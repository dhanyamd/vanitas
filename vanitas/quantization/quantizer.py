# vanitas/quantization/quantizer.py
"""Utility functions for quantizing the Vanitas model.

We provide three main workflows:
1️⃣ Static post‑training quantization (PTQ) – fast, no extra training.
2️⃣ Quantization‑aware training (QAT) – a short fine‑tune to recover any loss.
3️⃣ Weight‑only low‑bit quantization (e.g., 4‑bit) – extreme model size reduction.

All functions operate on a ``torch.nn.Module`` that follows the standard
VanitasModel API (i.e. ``model(inputs)`` returns a dict with the various
streams). The helper ``get_calib_loader`` in ``calib_loader.py`` yields raw
audio tensors for calibration.
"""

import torch
from torch import nn
from torch.ao.quantization import (
    get_default_qconfig,
    prepare,
    convert,
    prepare_qat,
)

# Optional import for weight‑only 4‑bit quantization (requires torch>=2.4 and torchao)
try:
    from torchao.quantization import quantize_weight_only
except Exception:  # pragma: no cover – torchao may not be installed yet
    quantize_weight_only = None


def quantize_static(model: nn.Module, calib_loader, dtype=torch.qint8) -> nn.Module:
    """Static post‑training quantization.

    Parameters
    ----------
    model: nn.Module
        The pretrained Vanitas model (in eval mode).
    calib_loader: ``torch.utils.data.DataLoader``
        Yields representative inputs (raw PCM float32 tensors) for the
        observer calibration step.
    dtype: torch.dtype, optional
        The target integer datatype for the quantized weights. The default
        ``torch.qint8`` matches the typical "int8" deployment.

    Returns
    -------
    nn.Module
        A quantized model ready for inference.
    """
    model.eval()
    # Attach a default per‑channel int8 qconfig (suitable for server‑side
    # inference on x86/ARM CPUs).
    qconfig = get_default_qconfig("fbgemm")
    model.qconfig = qconfig

    # Insert observers (FakeQuant) into the model.
    prepared = prepare(model)

    # Calibration – run a few batches through the model to collect statistics.
    with torch.no_grad():
        for batch in calib_loader:
            # ``batch`` may contain tuples; we assume the first element is the
            # raw audio tensor expected by the model.
            inputs = batch[0]
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]
            inputs = inputs.to(next(prepared.parameters()).device)
            prepared(inputs)

    # Convert observers -> real int8 kernels.
    quantized = convert(prepared, dtype=dtype)
    return quantized


def quantize_qat(
    model: nn.Module,
    train_loader,
    epochs: int = 1,
    lr: float = 1e-4,
    loss_fn: torch.nn.Module = None,
) -> nn.Module:
    """Quantization‑aware training.

    This runs a short fine‑tune on the same data you used for the last joint
    training pass. ``loss_fn`` should be the same loss you used for the original
    model (e.g., the combined BCE + contrastive + flow‑matching loss). If not
    supplied we fall back to an MSE loss on the model output tensor – this is a
    safe default for a quick sanity check.
    """
    model.train()
    qconfig = get_default_qconfig("fbgemm")
    model.qconfig = qconfig
    qat_model = prepare_qat(model)

    optimiser = torch.optim.Adam(qat_model.parameters(), lr=lr)
    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()

    device = next(qat_model.parameters()).device
    for epoch in range(epochs):
        for batch in train_loader:
            optimiser.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = qat_model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimiser.step()

    # Convert fake‑quant modules into real int8 kernels.
    quantized = convert(qat_model)
    return quantized


def quantize_weight_only_4bit(model: nn.Module) -> nn.Module:
    """Weight‑only 4‑bit quantization.

    Activations remain in FP16 (or BF16) while the linear weights are stored in
    4‑bit integer format with per‑group scaling. This yields the smallest model
    footprint with only a modest latency impact.
    """
    if quantize_weight_only is None:
        raise RuntimeError(
            "torchao is not available. Install with `pip install 'torch[ao]'` "
            "to use weight‑only low‑bit quantization."
        )
    # Quantize every Linear layer; you can extend to Embedding if desired.
    qo_model = quantize_weight_only(
        model,
        weight_dtype=torch.int8,  # 4‑bit is internally stored in int8 with scaling
        groupsize=128,
        target_modules=("Linear",),
    )
    return qo_model
