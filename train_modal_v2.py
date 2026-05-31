"""Vanitas-SLM v2 — Modal entrypoint for cloud training.

Usage (Stage 1, full):
    modal run --detach train_modal_v2.py::stage1

Usage (Stage 1, smoke test on a small instance first):
    modal run train_modal_v2.py::stage1_smoke

Stages 2-6 will be wired in as their training scripts land.
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal


# ---------------------------------------------------------------------------
# Image — PyTorch CUDA base with Vanitas-SLM v2 dependencies
# ---------------------------------------------------------------------------

vanitas_v2_image = (
    # PyTorch 2.4 / CUDA 12.4 base. We deliberately do NOT install unsloth here
    # because its torch>=2.4,<2.11 resolver will upgrade torch to 2.10 and
    # break torchvision's C++ ops binding (torchvision::nms missing). For QLoRA
    # in stages 2+ we use plain peft + bitsandbytes, which is slower than
    # unsloth's fused kernels but doesn't require a torch version bump.
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")
    .apt_install("libsndfile1", "git", "build-essential")
    .run_commands(
        "pip install -q --upgrade pip wheel setuptools",
        # Core deps. transformers pinned to 4.x because the 5.x line ships
        # behavioural changes we haven't verified against our HF-layer code.
        "pip install "
        "'numpy>=1.26.0' "
        "'scipy>=1.12.0' "
        "'einops>=0.8.0' "
        "'transformers>=4.45.0,<5.0' "
        "'datasets>=3.0.0' "
        "'huggingface_hub[hf_transfer]>=0.20.0' "
        "'accelerate>=0.30.0' "
        "'bitsandbytes>=0.43.0' "
        "'peft>=0.11.0' ",
        # Mamba-2 official kernels. --no-build-isolation lets it find the
        # already-installed torch from the base image instead of pulling a new one.
        "pip install --no-build-isolation 'mamba-ssm>=2.2.0' 'causal-conv1d>=1.4.0'",
        # SNAC codec.
        "pip install snac",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(
        str(Path(__file__).resolve().parent / "vanitas"),
        remote_path="/root/vanitas",
    )
)


app = modal.App(name="vanitas-slm-v2")

# Persistent volumes that survive between runs.
checkpoints_volume = modal.Volume.from_name("vanitas-slm-checkpoints", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("vanitas-slm-hf-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# Stage 1 — full run
# ---------------------------------------------------------------------------

@app.function(
    image=vanitas_v2_image,
    gpu="A100",
    timeout=6 * 3600,  # 6h cap; Stage 1 should finish well under this.
    volumes={
        "/root/checkpoints": checkpoints_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def stage1(
    total_steps: int = 3000,
    batch_size: int = 4,
    grad_accum: int = 64,
    seq_len: int = 2048,
    lr: float = 5e-5,
    warmup: int = 200,
    eval_every: int = 500,
):
    """Stage 1 — Mamba surgery + distillation. PLAN.md §3 Stage 1."""
    import os
    import subprocess

    sys.path.insert(0, "/root")
    os.makedirs("/root/checkpoints/stage1", exist_ok=True)

    cmd = [
        "python", "-m", "vanitas.v2.training.stage1_mamba_distill",
        "--total-steps", str(total_steps),
        "--batch-size", str(batch_size),
        "--grad-accum", str(grad_accum),
        "--seq-len", str(seq_len),
        "--lr", str(lr),
        "--warmup", str(warmup),
        "--eval-every", str(eval_every),
        "--out-dir", "/root/checkpoints/stage1",
    ]
    print("⇒", " ".join(cmd))
    result = subprocess.run(cmd, cwd="/root")
    checkpoints_volume.commit()
    hf_cache_volume.commit()
    if result.returncode != 0:
        raise SystemExit(f"Stage 1 exited with code {result.returncode}")


@app.function(
    image=vanitas_v2_image,
    gpu="A10G",  # Smaller cheaper GPU is fine for the smoke pass.
    timeout=30 * 60,
    volumes={
        "/root/checkpoints": checkpoints_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def stage1_smoke():
    """5-step smoke test on a cheap GPU to confirm the pipeline runs end-to-end."""
    import subprocess
    sys.path.insert(0, "/root")
    cmd = ["python", "-m", "vanitas.v2.training.stage1_mamba_distill", "--smoke",
           "--out-dir", "/root/checkpoints/stage1_smoke"]
    print("⇒", " ".join(cmd))
    result = subprocess.run(cmd, cwd="/root")
    checkpoints_volume.commit()
    hf_cache_volume.commit()
    if result.returncode != 0:
        raise SystemExit(f"Stage 1 smoke exited with code {result.returncode}")


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Default entrypoint: prints recommended commands and exits.

    There is intentionally no default training run here. The smoke test costs
    a couple of dollars; the full Stage 1 costs ~$80. We do not want to launch
    either accidentally just because someone typed ``modal run train_modal_v2.py``.

    Recommended sequence:
        modal run train_modal_v2.py::stage1_smoke      (~$2, ~10 min)
        modal run --detach train_modal_v2.py::stage1   (~$80, 3-4 hrs)
    """
    print(
        "──────────────────────────────────────────────────────────────────\n"
        "  Vanitas-SLM v2 — Modal entrypoint\n"
        "──────────────────────────────────────────────────────────────────\n"
        "  No default action. Pick one explicitly:\n\n"
        "    modal run train_modal_v2.py::stage1_smoke\n"
        "        Cheap (~$2) 5-step smoke test on an A10G. Confirms the\n"
        "        whole pipeline runs end-to-end on CUDA before committing\n"
        "        to the full Stage 1 spend.\n\n"
        "    modal run --detach train_modal_v2.py::stage1\n"
        "        Full Stage 1 on A100 (~$80, 3-4 hrs). --detach is\n"
        "        recommended so the run survives terminal disconnects.\n"
        "──────────────────────────────────────────────────────────────────"
    )
