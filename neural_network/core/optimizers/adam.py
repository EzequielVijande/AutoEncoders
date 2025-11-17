"""
Adam optimizer.
"""

import numpy as np
from .base import OptimizerFunction
from ..network import NeuralNetwork


class AdamOptimizer(OptimizerFunction):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    
    Combines the advantages of AdaGrad and RMSprop by computing adaptive
    learning rates for each parameter from estimates of first and second
    moments of the gradients.
    """
    
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)

        # State variables
        self.m = {}  # First moment vector
        self.v = {}  # Second moment vector
        self.timestep = {}

    def update_network(self, network: NeuralNetwork, learning_rate: float):
        """
        Update all weights in the network using Adam.
        `gradients` should be a list of lists: gradients[layer][perceptron]
        """
        for layer_idx, layer in enumerate(network.layers):
            if hasattr(layer, 'layers'):  # If the layer is a composite layer (like StochasticLayer)
                for l in layer.layers:
                    key = id(l)
                    if key not in self.m:
                        self.m[key] = np.zeros_like(l.weights, dtype=np.float32)
                        self.v[key] = np.zeros_like(l.weights, dtype=np.float32)
                        self.timestep[key] = 0

                    self.timestep[key] += 1
                    self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * l.grad
                    self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (l.grad ** 2)
                    m_hat = self.m[key] / (1 - self.beta1 ** self.timestep[key])
                    v_hat = self.v[key] / (1 - self.beta2 ** self.timestep[key])
                    l.weights = l.weights - learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                    l.grad = 0.0
                continue
            else:
                key = id(layer)

                if key not in self.m:
                    self.m[key] = np.zeros_like(layer.weights, dtype=np.float32)
                    self.v[key] = np.zeros_like(layer.weights, dtype=np.float32)
                    self.timestep[key] = 0

                self.timestep[key] += 1
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * layer.grad
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (layer.grad ** 2)
                m_hat = self.m[key] / (1 - self.beta1 ** self.timestep[key])
                v_hat = self.v[key] / (1 - self.beta2 ** self.timestep[key])
                layer.weights = layer.weights - learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                layer.grad = 0.0
    
    
    def reset_state(self) -> None:
        """Reset optimizer state."""
        self.m = {}
        self.v = {}
        self.timestep = {}