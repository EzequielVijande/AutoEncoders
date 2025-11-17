"""
Base class for optimizer functions.
"""

from abc import ABC, abstractmethod
import numpy as np
from ..network import NeuralNetwork


class OptimizerFunction(ABC):
    
    @abstractmethod
    def update_network(self, network: NeuralNetwork, learning_rate: float):
        pass
    
    @abstractmethod
    def reset_state(self) -> None:
        pass