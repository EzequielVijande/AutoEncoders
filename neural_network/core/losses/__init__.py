"""
Loss functions for neural networks.
"""

from .functions import softmax_cross_entropy_with_logits, mae, mse

__all__ = [
    'softmax_cross_entropy_with_logits',
    'mae',
    'mse'
]