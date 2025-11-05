import sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from neural_network.utils.data_utils import parse_font_file
from neural_network.core.network import NeuralNetwork
from neural_network.core.trainer import Trainer
from neural_network.core.losses.functions import mse
from neural_network.config import OptimizerConfig

DATASET_PATH = "resources/datasets/font.h"


def main():
    #Cargar  parsear dataset
    dst = parse_font_file(DATASET_PATH)
    X = np.zeros((len(dst), len(dst[list(dst.keys())[0]])))
    for i, key in enumerate(dst):
        X[i] = dst[key]
    #Inicializar modelo AE
    topology = [35, 26, 10, 26, 35]
    activations = ['relu', 'relu', 'relu', 'sigmoid']
    nn = NeuralNetwork(topology, activations)
    #Parametros de entrenamiento
    b_size = 32
    lr = 1e-3
    epochs = 10000
    opt_cfg = OptimizerConfig("ADAM")
    tr = Trainer(lr, epochs, nn, mse, opt_cfg)
    #Entrenar modelo
    tr_losses, _ = tr.train(X, X, b_size)
    #Graficar perdida
    plt.figure()
    plt.plot(tr_losses, label="Pérdida de Entrenamiento")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.title("Pérdida durante el Entrenamiento del Autoencoder")
    plt.show()

    pred = tr.network.forward(X[2].reshape((1,-1)), training=False)
    plt.figure()
    plt.title('Original')
    plt.imshow(X[2].reshape((7,5)), cmap='gray_r')
    plt.show()
    plt.figure()
    plt.title('Reconstruido')
    plt.imshow(pred.reshape((7,5)), cmap='gray_r')
    plt.show()






if __name__ == "__main__":
    from experiments.font_encoder.main import main

    main()