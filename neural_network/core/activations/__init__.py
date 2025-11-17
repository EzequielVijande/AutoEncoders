"""
Activation functions for neural networks.

This module provides a flexible system for activation functions using the Strategy pattern.
Each activation function is implemented in its own file and can be created via the factory.
"""

# Core interfaces
from .base import ActivationFunction

# Individual activation functions
from .step import StepActivation
from .step_bipolar import StepBipolarActivation
from .sigmoid import SigmoidActivation
from .relu import ReLUActivation
from .tanh import TanhActivation

# Factory for creating activation functions
from .factory import ActivationFunctionFactory, ActivationFunctionType

# Exported symbols
__all__ = [
    # Core interfaces
    "ActivationFunction",
    "ActivationFunctionType",
    
    # Concrete implementations
    "StepActivation",
    "StepBipolarActivation", 
    "SigmoidActivation",
    "ReLUActivation",
    
    # Factory
    "ActivationFunctionFactory",
]