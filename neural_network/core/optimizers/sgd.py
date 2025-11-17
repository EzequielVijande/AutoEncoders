"""
Stochastic Gradient Descent optimizer.
"""

import numpy as np
from .base import OptimizerFunction
from ..network import NeuralNetwork
from typing import List, Union, Tuple


class SGDOptimizer(OptimizerFunction):
    """
    Standard Stochastic Gradient Descent optimizer.
    
    Performs simple gradient descent: weights = weights - learning_rate * gradients
    """
    
    def __init__(self):
        """Initialize SGD optimizer."""
        pass
    
    def update_network(self, network: NeuralNetwork, learning_rate: float):
        """
        Update all weights in the network using SGD.
        `gradients` should be a list of lists: gradients[layer][perceptron]
        """
        for layer_idx, layer in enumerate(network.layers):
            if hasattr(layer, 'layers'):  # If the layer is a composite layer (like StochasticLayer)
                for l in layer.layers:
                    l.weights = l.weights - learning_rate * l.grad
            else:
                layer.weights = layer.weights - learning_rate * layer.grad
                # grad = np.array(gradients[layer_idx]).transpose()  # Shape: (input_dim + 1, num_perceptrons)
                # network.layers[layer_idx].weights = layer.weights - learning_rate * grad
                   
    def reset_state(self) -> None:
        pass