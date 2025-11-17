"""
Tanh activation function implementation.
"""

import numpy as np
from .base import ActivationFunction


class TanhActivation(ActivationFunction):
    """
    Tangent Hyperbolic activation function.

    Formula: f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Range: (-1, 1)

    Advantages:
    - Zero-centered output (mean near 0)
    - Symmetric around origin
    - Smooth gradient
    - Prevents bias shift in next layer
    """

    @property
    def name(self) -> str:
        """Return the name of the activation function."""
        return "Tanh"

    @property
    def output_range(self) -> tuple:
        """Return the output range of the activation function."""
        return (-1.0, 1.0)

    def activate(self, x: np.ndarray) -> np.ndarray:
        """
        Apply tanh activation.

        Args:
            x: Input array

        Returns:
            Output in range (-1, 1)
        """
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute derivative of tanh.

        Formula: f'(x) = 1 - tanh²(x)

        Args:
            x: Input array (pre-activation values)

        Returns:
            Derivative values
        """
        tanh_x = np.tanh(x)
        return 1 - tanh_x ** 2

