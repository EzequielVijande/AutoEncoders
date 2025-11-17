from neural_network.core.layer import Layer
import numpy as np


class StochasticLayer(Layer):
    def __init__(self, num_perceptrons: int, num_inputs_per_perceptron: int,
                activation_type: str = "LINEAR", dropout_rate: float = 0.0):
        self.name = "STOCHASTIC"
        self.dropout_rate: float = dropout_rate
        self.mean_layer = Layer(num_perceptrons, num_inputs_per_perceptron,
                            "LINEAR", dropout_rate)
        self.sigma_layer = Layer(num_perceptrons, num_inputs_per_perceptron,
                            "LINEAR", dropout_rate)
        self.layers = [self.mean_layer, self.sigma_layer]


    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:

        #Calculate parameters and sample
        self.location = self.mean_layer.forward(inputs)#np.dot(inputs_with_bias, self.mean_layer.weights)    # Shape: (batch_size, num_perceptrons)
        self.scale = self.sigma_layer.forward(inputs)#np.dot(inputs_with_bias, self.sigma_layer.weights)    # Shape: (batch_size, num_perceptrons)
        samples = np.random.normal(loc=0.0, scale=1.0, size=self.scale.shape)
        self.last_sample = samples
        outputs = samples * np.exp(self.scale/2) + self.location #Sample using reparametrization trick

        #Apply dropout if set
        if training and self.dropout_rate > 0.0:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=outputs.shape).astype(np.float32)
            outputs *= self.mask
        else:
            outputs *= (1 - self.dropout_rate)

        return outputs
    
    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        # Gradients w.r.t. location and scale
        grad_location = output_gradients  # Shape: (batch_size, num_perceptrons)
        grad_scale = output_gradients * self.last_sample *0.5*np.exp(self.scale/2)  # Shape: (batch_size, num_perceptrons)

        # Backprop through mean and sigma layers
        grad_inputs_mean = self.mean_layer.backward(grad_location)  # Shape: (batch_size, num_inputs + 1)
        grad_inputs_sigma = self.sigma_layer.backward(grad_scale)    # Shape: (batch_size, num_inputs + 1)

        # Total gradient w.r.t. inputs
        total_grad_inputs = grad_inputs_mean + grad_inputs_sigma  # Shape: (batch_size, num_inputs + 1)

        return total_grad_inputs  # Shape: (batch_size, num_inputs)