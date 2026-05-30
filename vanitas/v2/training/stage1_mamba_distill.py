"""Stage 1 — Mamba surgery + distillation back to base Qwen3 (PLAN.md §3 Stage 1).

What this script does, top to bottom:

  1. Loads base Qwen3-1.7B as a frozen **teacher**.
  2. Loads another copy as a **student**, inserts 4 Mamba-2 blocks at
     positions 6/13/20/27 (zero-init residual gate ⇒ initially identity),
     leaves the vocab at its base size (vocab expansion happens in Stage 2).
  3. Streams text from OpenWebText, tokenizes to (B, L=2048) blocks.
  4. Forward both models on the same batch; compute
     ``KL(student || teacher)`` over the vocab as the only loss.
  5. AdamW + cosine schedule + grad-accum to an effective batch ≥256.
  6. Periodically evals validation perplexity on WikiText-103.
     **Exit criterion:** val-PPL within 5 % of base Qwen3 PPL on the
     same prompts. If not met, the run is failed.

CUDA only. The script imports cleanly on Mac (so you can read / lint it),
but the training calls will raise as soon as Mamba surgery is attempted.

Usage (smoke test):
    python -m vanitas.v2.training.stage1_mamba_distill --smoke

Usage (full Stage 1):
    python -m vanitas.v2.training.stage1_mamba_distill \\
        --total-steps 3000 --batch-size 4 --grad-accum 64 \\
        --lr 5e-5 --warmup 200 --seq-len 2048 \\
        --out-dir checkpoints/v2/stage1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset


# ---------------------------------------------------------------------------
# Dataset: streaming OpenWebText, packed to fixed-length sequences
# ---------------------------------------------------------------------------

class PackedTextStream(IterableDataset):
    """Streams text from HF datasets, tokenizes, and emits packed sequences.

    We pack documents end-to-end with EOS in between so every yielded
    sequence is exactly ``seq_len`` tokens. No padding, no wasted compute.
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int = 2048,
        dataset_name: str = "Skylion007/openwebtext",
        split: str = "train",
        text_column: str = "text",
        max_tokens: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.split = split
        self.text_column = text_column
        self.max_tokens = max_tokens

    def __iter__(self):
        from datasets import load_dataset

        ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
        eos = self.tokenizer.eos_token_id
        buf: list[int] = []
        seen_tokens = 0

        for sample in ds:
            text = sample.get(self.text_column) or ""
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            buf.extend(ids)
            if eos is not None:
                buf.append(eos)

            while len(buf) >= self.seq_len:
                seq = buf[: self.seq_len]
                buf = buf[self.seq_len:]
                seen_tokens += self.seq_len
                yield torch.tensor(seq, dtype=torch.long)

                if self.max_tokens is not None and seen_tokens >= self.max_tokens:
                    return


# ---------------------------------------------------------------------------
# Evaluation: PPL on a small held-out slice
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ppl(
    model: nn.Module,
    tokenizer,
    max_eval_tokens: int = 200_000,
    seq_len: int = 2048,
    device: str | torch.device = "cuda",
    dataset_name: str = "wikitext",
    config_name: str | None = "wikitext-103-raw-v1",
    split: str = "validation",
) -> float:
    """Token-level perplexity on a fixed slice of WikiText-103 validation."""
    from datasets import load_dataset

    model.eval()
    ds = load_dataset(dataset_name, config_name, split=split, streaming=True)

    eos = tokenizer.eos_token_id
    buf: list[int] = []
    total_loss = 0.0
    total_tokens = 0

    for sample in ds:
        text = sample.get("text") or ""
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)
        if eos is not None:
            buf.append(eos)

        while len(buf) >= seq_len:
            seq = buf[:seq_len]
            buf = buf[seq_len:]
            x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x).logits
            # Standard next-token shift
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                x[:, 1:].reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_tokens += x.numel() - 1
            if total_tokens >= max_eval_tokens:
                model.train()
                avg_nll = total_loss / total_tokens
                return float(torch.tensor(avg_nll).exp().item())

    model.train()
    avg_nll = total_loss / max(1, total_tokens)
    return float(torch.tensor(avg_nll).exp().item())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        print(
            "❌ Stage 1 training requires CUDA. mamba-ssm has no CPU/MPS kernels.\n"
            "   Run this on a Modal/Lambda/RunPod A100/H100/4090 box.",
            file=sys.stderr,
        )
        return 2

    device = torch.device("cuda")
    print(f"[Stage 1] device={device.type}:{torch.cuda.current_device()}  "
          f"name={torch.cuda.get_device_name(0)}  "
          f"vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # ---- Models ----------------------------------------------------------
    from vanitas.v2.model.backbone import (
        DEFAULT_MAMBA_POSITIONS,
        apply_mamba_surgery,
        load_base_qwen3,
    )
    from vanitas.v2.training.common import (
        StepLogger,
        TrainingManifest,
        cosine_warmup_schedule,
        kl_distillation_loss,
        load_qwen3_for_training,
        save_training_checkpoint,
    )

    print("[Stage 1] Loading frozen teacher (base Qwen3-1.7B, bf16)...")
    teacher = load_base_qwen3(dtype=torch.bfloat16, device=device)

    print("[Stage 1] Loading student (copy of teacher) + applying Mamba surgery...")
    student, tokenizer = load_qwen3_for_training(use_qlora=False, dtype=torch.bfloat16)
    student.to(device)
    apply_mamba_surgery(student, positions=DEFAULT_MAMBA_POSITIONS)
    student.train()
    student.gradient_checkpointing_enable()

    # Freeze everything except the new Mamba blocks (and their layer-norm + gate).
    # The base Qwen3 weights should not move during Stage 1 — the entire job is
    # to teach the SSM layers to play nice in-residual.
    trainable_params: list[nn.Parameter] = []
    for name, p in student.named_parameters():
        is_mamba_layer = any(
            f".layers.{pos}." in name for pos in DEFAULT_MAMBA_POSITIONS
        )
        p.requires_grad_(is_mamba_layer)
        if is_mamba_layer:
            trainable_params.append(p)
    n_train = sum(p.numel() for p in trainable_params)
    print(f"[Stage 1] Trainable Mamba params: {n_train:,} (~{n_train/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.95)
    )

    # ---- Data ------------------------------------------------------------
    stream = PackedTextStream(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        max_tokens=args.total_steps * args.batch_size * args.seq_len * args.grad_accum + 100_000,
    )
    loader = DataLoader(stream, batch_size=args.batch_size, num_workers=0)
    loader_iter = iter(loader)

    # ---- Training loop ---------------------------------------------------
    logger = StepLogger(log_every=args.log_every, stage="Stage1")
    t0 = time.time()

    for step in range(1, args.total_steps + 1):
        # LR
        lr_mult = cosine_warmup_schedule(step, args.warmup, args.total_steps)
        for g in optimizer.param_groups:
            g["lr"] = args.lr * lr_mult

        accum_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for _ in range(args.grad_accum):
            try:
                batch = next(loader_iter)
            except StopIteration:
                # Restart stream if we run out (shouldn't normally happen with max_tokens math above)
                loader_iter = iter(loader)
                batch = next(loader_iter)
            batch = batch.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                student_logits = student(batch).logits
                with torch.no_grad():
                    teacher_logits = teacher(batch).logits

                loss = kl_distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    temperature=args.kd_temperature,
                ) / args.grad_accum

            loss.backward()
            accum_loss += float(loss.item()) * args.grad_accum

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        logger.log(step, accum_loss / args.grad_accum, args.lr * lr_mult)

        # Periodic eval + checkpoint
        if step % args.eval_every == 0 or step == args.total_steps:
            print(f"[Stage 1] Eval at step {step}...")
            student_ppl = evaluate_ppl(student, tokenizer, device=device)
            teacher_ppl = evaluate_ppl(teacher, tokenizer, device=device)
            ratio = student_ppl / teacher_ppl
            print(
                f"[Stage 1] PPL  student={student_ppl:.3f}  teacher={teacher_ppl:.3f}  "
                f"ratio={ratio:.3f}  (exit criterion: ratio ≤ 1.05)"
            )

            manifest = TrainingManifest(
                stage="stage1_mamba_distill",
                step=step,
                total_steps=args.total_steps,
                loss=accum_loss / args.grad_accum,
                val_metric=student_ppl,
                wallclock_s=time.time() - t0,
                config=vars(args),
            )
            save_training_checkpoint(
                out_dir=Path(args.out_dir) / f"step_{step}",
                model=student,
                optimizer=optimizer,
                manifest=manifest,
                tokenizer=tokenizer,
            )

            if step == args.total_steps:
                if ratio <= 1.05:
                    print("✅ Stage 1 EXIT CRITERION MET (ratio ≤ 1.05).")
                    return 0
                print(f"❌ Stage 1 EXIT CRITERION FAILED (ratio {ratio:.3f} > 1.05).")
                return 1

    return 0


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1 — Mamba surgery + distillation")
    parser.add_argument("--total-steps", type=int, default=3000,
                        help="Optimizer steps (after gradient accumulation).")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device micro-batch.")
    parser.add_argument("--grad-accum", type=int, default=64,
                        help="Gradient accumulation steps. Effective batch = batch_size * grad_accum.")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Token sequence length per micro-batch.")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--out-dir", type=str, default="checkpoints/v2/stage1")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a 5-step smoke pass; ignores other knobs.")
    args = parser.parse_args()

    if args.smoke:
        args.total_steps = 5
        args.batch_size = 2
        args.grad_accum = 2
        args.seq_len = 512
        args.eval_every = 5
        args.log_every = 1
        args.warmup = 1
        print("[Stage 1] Smoke mode: 5 steps, batch 2, grad-accum 2, seq 512.")

    return train(args)


if __name__ == "__main__":
    sys.exit(main())
