"""Inject Vanitas's Mamba-2 surgery into X-LANCE/SLAM-LLM without forking it.

SLAM-LLM builds the LLM backbone in ``slam_llm.models.slam_model.setup_llm``
(a plain ``AutoModelForCausalLM.from_pretrained``). Its s2s ``model_factory``
calls that, then wraps the result in ``slam_model_s2s``.

We monkeypatch ``setup_llm`` so that, right after the base Qwen3 is loaded, we
insert our validated Mamba-2 residual blocks (the same ``apply_mamba_surgery``
that passed the PPL-ratio-1.000 smoke test). SLAM-LLM's own source stays
untouched — this keeps the integration clean and reproducible: a reviewer runs
SLAM-LLM as-is plus this one import.

Configuration is via environment variables so we don't have to extend
SLAM-LLM's Hydra config schema:

    VANITAS_MAMBA_ENABLE=1            # turn the patch on (default off → no-op)
    VANITAS_MAMBA_POSITIONS=6,13,20,27   # insert-after layer indices
    VANITAS_MAMBA_DSTATE=128
    VANITAS_MAMBA_DCONV=4
    VANITAS_MAMBA_EXPAND=2

Usage (in the SLAM-LLM training entrypoint, before model_factory runs):

    from vanitas.v2.integration.slam_mamba_patch import patch_slam_llm
    patch_slam_llm()

or set VANITAS_MAMBA_ENABLE=1 and import this module — it self-applies on import.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("vanitas.integration.slam_mamba_patch")

_PATCHED = False


def _env_positions(default: tuple[int, ...] = (6, 13, 20, 27)) -> tuple[int, ...]:
    raw = os.environ.get("VANITAS_MAMBA_POSITIONS")
    if not raw:
        return default
    return tuple(int(x) for x in raw.split(",") if x.strip() != "")


def _enabled() -> bool:
    return os.environ.get("VANITAS_MAMBA_ENABLE", "0").lower() in {"1", "true", "yes"}


def patch_slam_llm(force: bool = False) -> bool:
    """Monkeypatch SLAM-LLM's ``setup_llm`` to apply Mamba surgery.

    Returns True if the patch was installed, False if skipped (disabled or
    already patched). Safe to call multiple times.
    """
    global _PATCHED
    if _PATCHED and not force:
        return False
    if not _enabled():
        logger.info("VANITAS_MAMBA_ENABLE not set; Mamba surgery patch is a no-op.")
        return False

    try:
        import slam_llm.models.slam_model as slam_model_mod
    except ImportError as exc:
        raise ImportError(
            "Could not import slam_llm.models.slam_model. Run inside the SLAM-LLM "
            "environment (it must be on PYTHONPATH)."
        ) from exc

    from vanitas.v2.model.backbone import apply_mamba_surgery

    positions = _env_positions()
    d_state = int(os.environ.get("VANITAS_MAMBA_DSTATE", "128"))
    d_conv = int(os.environ.get("VANITAS_MAMBA_DCONV", "4"))
    expand = int(os.environ.get("VANITAS_MAMBA_EXPAND", "2"))

    original_setup_llm = slam_model_mod.setup_llm

    def patched_setup_llm(train_config, model_config, **kwargs):
        model = original_setup_llm(train_config, model_config, **kwargs)
        try:
            n_before = sum(p.numel() for p in model.parameters())
            apply_mamba_surgery(
                model,
                positions=positions,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            n_after = sum(p.numel() for p in model.parameters())
            logger.info(
                "[vanitas] Mamba surgery applied: inserted %d blocks after layers %s. "
                "Params %d -> %d (+%d).",
                len(positions), positions, n_before, n_after, n_after - n_before,
            )
        except Exception:
            logger.exception("[vanitas] Mamba surgery FAILED; aborting so we don't "
                             "silently train a vanilla model.")
            raise
        return model

    slam_model_mod.setup_llm = patched_setup_llm
    _PATCHED = True
    logger.info(
        "[vanitas] Patched slam_llm setup_llm. positions=%s d_state=%d d_conv=%d expand=%d",
        positions, d_state, d_conv, expand,
    )
    return True


# Self-apply on import when enabled, so a single `import` in the training script
# is enough (no code change inside SLAM-LLM).
if _enabled():
    try:
        patch_slam_llm()
    except Exception:  # pragma: no cover - import-time best effort
        logger.exception("[vanitas] setup_llm patch failed at import time.")
