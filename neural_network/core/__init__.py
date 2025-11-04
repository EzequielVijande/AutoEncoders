"""
Core components of the neural network framework.
"""

from .network import NeuralNetwork
from .layer import Layer  

# Re-export activation and optimizer components for convenience
from .activations import ActivationFunction, ActivationFunctionFactory
from .optimizers import OptimizerFunction, OptimizerFunctionFactory

__all__ = [
    "NeuralNetwork", 
    "Layer", 
    "ActivationFunction",
    "ActivationFunctionFactory", 
    "OptimizerFunction",
    "OptimizerFunctionFactory"
]