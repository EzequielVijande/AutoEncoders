"""
Neural Network Framework
========================

Una implementación modular de redes neuronales desde cero.
"""

from .core.network import NeuralNetwork
from .core.layer import Layer

__version__ = "1.0.0"
__all__ = ["NeuralNetwork", "Layer"]