# exp.py

import numpy as np
from .base import ActivationFunction


class ExpActivation(ActivationFunction):
    """
    Función de activación Exponencial.
    
    Salida: f(x) = e(x/2)
    Rango: [0, +∞)
    """
    
    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x/2)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x/2)/2

    @property
    def name(self) -> str:
        return "Exp"

    @property
    def output_range(self) -> tuple:
        return (0.0, float('inf'))