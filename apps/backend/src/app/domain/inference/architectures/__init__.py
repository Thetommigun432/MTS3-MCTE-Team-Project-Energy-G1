"""
Model architectures for NILM inference.
"""
from .hybrid_cnn_transformer import HybridCNNTransformerAdapter
from .wavenilm_v3 import WaveNILM_v3

__all__ = ["HybridCNNTransformerAdapter", "WaveNILM_v3"]
