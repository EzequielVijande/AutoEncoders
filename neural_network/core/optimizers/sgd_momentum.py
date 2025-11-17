"""
SGD with momentum optimizer.
"""

import numpy as np
from .base import OptimizerFunction
from ..network import NeuralNetwork


class SGDMomentumOptimizer(OptimizerFunction):
    """
    Stochastic Gradient Descent with momentum optimizer.
    
    Accumulates velocity over time to help navigate past local minima
    and accelerate convergence in consistent directions.
    """
    
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.velocity_dict = {}
    
    def update_network(self, network: NeuralNetwork, learning_rate: float):
        """
        Update all weights in the network using SGD with momentum.
        `gradients` should be a list of lists: gradients[layer][perceptron]
        """        
        for layer_idx, layer in enumerate(network.layers):
            if hasattr(layer, 'layers'):  # If the layer is a composite layer (like StochasticLayer)
                for l in layer.layers:
                    key = id(l)
                    if key not in self.velocity_dict:
                        self.velocity_dict[key] = np.zeros_like(l.weights, dtype=np.float32)
                    self.velocity_dict[key] = self.momentum * self.velocity_dict[key] + l.grad
                    l.weights = l.weights - learning_rate * self.velocity_dict[key]
                    l.grad = 0.0
                continue
            else:
                key = id(layer)
                if key not in self.velocity_dict:
                    self.velocity_dict[key] = np.zeros_like(layer.weights, dtype=np.float32)
                self.velocity_dict[key] = self.momentum * self.velocity_dict[key] + layer.grad
                layer.weights = layer.weights - learning_rate * self.velocity_dict[key]
                layer.grad = 0.0
    
    def reset_state(self) -> None:
        self.velocity = None