# layer.py

import numpy as np
from .activations import ActivationFunction, ActivationFunctionFactory
class Layer:
    def __init__(self, num_perceptrons: int, num_inputs_per_perceptron: int,
                 activation_type: str = "SIGMOID",
                 dropout_rate: float = 0.0) -> None:
        self.name = "FULLY CONNECTED"
        # Weight matrix: (num_inputs + 1, num_perceptrons), last row is bias
        self.weights = np.random.randn(num_inputs_per_perceptron + 1, num_perceptrons) * np.sqrt(2/(num_inputs_per_perceptron + 1 +num_perceptrons))
        self.activation_type = activation_type
        self.activation: ActivationFunction = ActivationFunctionFactory.create(activation_type)
        self.outputs: np.ndarray = np.array([])
        self.dropout_rate: float = dropout_rate
        self.mask: np.ndarray = np.array([])
        self.last_z: np.ndarray = np.array([])
        self.last_inputs: np.ndarray = np.array([])
        self.grad = 0.0

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
    
    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        # Apply dropout mask to output gradients if dropout was used
        if hasattr(self, 'mask') and self.mask.size > 0:
            output_gradients *= self.mask

        activation_derivatives = self._get_activation_derivative()  # Shape: (batch_size, num_perceptrons)
        delta = output_gradients * activation_derivatives  # Shape: (batch_size, num_perceptrons)

        # Gradient w.r.t. weights
        weight_gradients = np.dot(self.last_inputs.T, delta) / self.last_inputs.shape[0]  # Shape: (num_inputs + 1, num_perceptrons)

        # Gradient w.r.t. inputs to propagate to previous layer
        input_gradients = np.dot(delta, self.weights[:-1, :].T)  # Shape: (batch_size, num_inputs)

        # Update weights - this will be handled by the optimizer in practice
        self.grad += weight_gradients

        return input_gradients

    def _get_activation_derivative(self) -> np.ndarray:
        derivatives = self.activation.derivative(self.last_z)
        if hasattr(self, 'mask') and self.mask.size > 0:
            derivatives *= self.mask
        return derivatives
