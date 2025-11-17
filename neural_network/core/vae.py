from neural_network.core.network import NeuralNetwork
from neural_network.core.auto_encoder import AutoEncoder
from neural_network.core.layer import Layer
from neural_network.core.stochastic_layer import StochasticLayer
from typing import List, Union, Tuple
import numpy as np

class VAE(AutoEncoder):
    def __init__(self,
            topology: List[int],
            enc_activations: Union[List[str], str] = "SIGMOID",
            dec_activations: Union[List[str], str] = "SIGMOID",
            dropout_rate: float = 0.0):
        if len(topology) < 3:
            raise ValueError("La topología del AutoEncoder debe tener al menos tres capas (entrada, codificación y salida).")
        if isinstance(enc_activations, list):
            if len(enc_activations) != len(topology) - 2:
                raise ValueError("La lista de tipos de activación del codificador debe coincidir con el número de capas ocultas y de salida.")
            if isinstance(dec_activations, list):
                if len(dec_activations) != len(topology) - 1:
                    raise ValueError("La lista de tipos de activación del decodificador debe coincidir con el número de capas ocultas y de salida.")
        self.encoder = NeuralNetwork(topology[:-1], enc_activations, dropout_rate)
        self.stochastic_layer = StochasticLayer(topology[-1], topology[-2])
        self.decoder = NeuralNetwork(topology[::-1], dec_activations, dropout_rate)
        #Initialize neural netwok data members
        self.layers: List[Layer] = [*self.encoder.layers, self.stochastic_layer, *self.decoder.layers]
        self.activation_type: Union[List[str], str] = [*enc_activations, *dec_activations[::-1]] if isinstance(enc_activations, list) else enc_activations
        self.dropout_rate: float = dropout_rate

    def encode(self, inputs, training: bool = False) -> np.ndarray:
        params = self.encoder.forward(inputs, training)
        z = self.stochastic_layer.forward(params, training)
        return z
    
    def decode(self, encoded_inputs, training: bool = False) -> np.ndarray:
        return self.decoder.forward(encoded_inputs, training)
    
    def get_params(self, inputs) -> Tuple[np.ndarray, np.ndarray]:
        params_in = self.encoder.forward(inputs, False)
        self.stochastic_layer.forward(params_in, False)
        return (self.stochastic_layer.location, self.stochastic_layer.scale)
    
    def forward(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        encoded = self.encode(inputs, training)
        decoded = self.decode(encoded, training)
        return decoded