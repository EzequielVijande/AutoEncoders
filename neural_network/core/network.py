# neural_network.py

import numpy as np
import os
from typing import List, Union
from .layer import Layer


class NeuralNetwork:
    def __init__(
            self,
            topology: List[int],
            activation_type: Union[List[str], str] = "SIGMOID",
            dropout_rate: float = 0.0
    ) -> None:
        if len(topology) < 2:
            raise ValueError("La topología debe tener al menos dos capas (entrada y salida).")
        if isinstance(activation_type, list):
            if len(activation_type) != len(topology) - 1:
                raise ValueError("La lista de tipos de activación debe coincidir con el número de capas ocultas y de salida.")

        self.layers: List[Layer] = []
        self.activation_type: Union[List[str], str] = activation_type
        self.dropout_rate: float = dropout_rate

        for i in range(1, len(topology)):
            # No aplicar dropout en la capa de salida
            layer_dropout_rate = dropout_rate if i < len(topology) - 1 else 0.0
            if isinstance(activation_type, list):
                layer_act = activation_type[i - 1].upper()
            else:
                layer_act = activation_type.upper()
            layer = Layer(
                num_perceptrons=topology[i],
                num_inputs_per_perceptron=topology[i - 1],
                activation_type=layer_act,
                dropout_rate=layer_dropout_rate
            )
            self.layers.append(layer)

    def get_topology(self) -> List[int]:
        topology = [self.layers[0].weights.shape[0]-1]  # Número de entradas (excluyendo el bias)
        for layer in self.layers:
            topology.append(layer.weights.shape[1])  # Número de perceptrones en la capa
        return topology

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if inputs.shape[1] != (self.layers[0].weights.shape[0]-1): #Se tiene en cuenta el bias
            raise ValueError("El número de características en las entradas no coincide con el esperado por la red.")
        for layer in self.layers:
            inputs = layer.forward(inputs, training=training)
        return inputs

    def evaluate(self, test_inputs: np.ndarray, test_labels: np.ndarray) -> float:
        if test_inputs.shape[0] != test_labels.shape[0]:
            raise ValueError("El número de muestras en 'test_inputs' y 'test_labels' debe ser el mismo.")

        logits = self.forward(test_inputs, training=False)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        predicted_classes = np.argmax(softmax_probs, axis=1)
        true_classes = np.argmax(test_labels, axis=1)
        accuracy = np.mean(predicted_classes == true_classes)
        return accuracy

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        logits = self.forward(inputs, training=False)

        # Aplicar softmax para obtener probabilidades
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return softmax_probs

    def save_weights(self, file_path: str) -> None:
        data = {'topology': self.get_topology(),
                'activation_type': self.activation_type,
                'dropout_rate': self.dropout_rate
                }
        for layer_num, layer in enumerate(self.layers):
            layer_weights = layer.weights
            data[f'layer_{layer_num}_weights'] = layer_weights
        np.savez(file_path, **data)
        print(f"Pesos guardados en '{file_path}'.")

    def load_weights(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo '{file_path}' no existe.")

        data = np.load(file_path, allow_pickle=True)
        num_layers = len(self.layers)
        for layer_num in range(num_layers):
            layer_weights = data[f'layer_{layer_num}_weights']
            self.layers[layer_num].weights = layer_weights
        print(f"Pesos cargados desde '{file_path}'.")

    @staticmethod
    def from_weights_file(file_path: str) -> 'NeuralNetwork':
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
        data = np.load(file_path, allow_pickle=True)
        topology = data['topology'].tolist()
        activation_type = data['activation_type'].item()  # Ya es string
        dropout_rate = data['dropout_rate'].item()  # Cargar el dropout_rate

        # Crear una nueva instancia de la red con la topología, activación y dropout cargados
        nn = NeuralNetwork(
            topology=topology,
            activation_type=activation_type,
            dropout_rate=dropout_rate  # Pasar el dropout_rate al constructor
        )

        # Cargar los pesos y biases
        num_layers = len(nn.layers)
        for layer_num in range(num_layers):
            layer_weights = data[f'layer_{layer_num}_weights']
            nn.layers[layer_num].weights = layer_weights
        print(f"Red neuronal creada y pesos cargados desde '{file_path}'.")
        return nn

