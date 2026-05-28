# vanitas/quantization/__init__.py
"""Quantization utilities package for Vanitas.
Exports the main helpers so they can be imported as
`from vanitas.quantization import quantize_static, quantize_qat, quantize_weight_only_4bit`.
"""

from .quantizer import quantize_static, quantize_qat, quantize_weight_only_4bit
