# layer.py

import numpy as np
from .activations import ActivationFunction, ActivationFunctionFactory
class Layer:
    def __init__(self, num_perceptrons: int, num_inputs_per_perceptron: int,
                 activation_type: str = "SIGMOID",
                 dropout_rate: float = 0.0) -> None:
        # Weight matrix: (num_inputs + 1, num_perceptrons), last row is bias
        self.weights = np.random.randn(num_inputs_per_perceptron + 1, num_perceptrons) * np.sqrt(2/(num_inputs_per_perceptron + 1 +num_perceptrons))
        self.activation_type = activation_type
        self.activation: ActivationFunction = ActivationFunctionFactory.create(activation_type)
        self.outputs: np.ndarray = np.array([])
        self.dropout_rate: float = dropout_rate
        self.mask: np.ndarray = np.array([])
        self.last_z: np.ndarray = np.array([])
        self.last_inputs: np.ndarray = np.array([])

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        # Add bias term to inputs
        bias = np.ones((inputs.shape[0], 1), dtype=inputs.dtype)
        inputs_with_bias = np.hstack([inputs, bias])  # Shape: (batch_size, num_inputs + 1)
        z = np.dot(inputs_with_bias, self.weights)    # Shape: (batch_size, num_perceptrons)
        outputs = self.activation.activate(z)
        if training and self.dropout_rate > 0.0:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=outputs.shape).astype(np.float32)
            outputs *= self.mask
        else:
            outputs *= (1 - self.dropout_rate)
        self.outputs = outputs
        self.last_inputs = inputs_with_bias  # Save for backprop
        self.last_z = z                      # Save for backprop
        return self.outputs

    def get_activation_derivative(self) -> np.ndarray:
        derivatives = self.activation.derivative(self.last_z)
        if hasattr(self, 'mask') and self.mask.size > 0:
            derivatives *= self.mask
        return derivatives
