import sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from neural_network.utils.data_utils import parse_font_file
from neural_network.core.network import NeuralNetwork
from neural_network.core.auto_encoder import AutoEncoder
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
    activations = ['relu']*(len(topology)-2) + ['sigmoid']
    nn = NeuralNetwork(topology, activations)
    ae = AutoEncoder(topology[:3], activations[:2], activations[2:])
    print(f'Encoder topology: {ae.encoder.get_topology()}')
    print(f'Decoder topology: {ae.decoder.get_topology()}')
    print((f'AE topology ={ae.get_topology()}'))
    print(f'Number of layers in encoder = {len(ae.encoder.layers)}')
    print(f'Number of layers in decoder = {len(ae.decoder.layers)}')
    print(f'Number of layers in AE = {len(ae.layers)}')
    #Parametros de entrenamiento
    b_size = 32 #Conjunto completo como batch
    lr = 2e-2
    epochs = 10000
    opt_cfg = OptimizerConfig("SGD")
    tr = Trainer(lr, epochs, ae, mse, opt_cfg)
    tr2 = Trainer(lr, epochs, nn, mse, opt_cfg)
    #Entrenar modelo
    tr_losses, _ = tr.train(X, X, b_size)
    tr_losses2, _ = tr2.train(X, X, b_size)
    #Verificar diferencias entre codificado-decodificado y forward
    encoded = ae.decode(ae.encode(X, training=False), training=False)
    encoded2 = ae.forward(X, training=False)
    print(f'difference between encoded-decoded and forward: {np.sum(np.abs(encoded - encoded2))}')
    #Graficar perdida
    plt.figure()
    plt.plot(tr_losses, label="AE", lw=3)
    plt.plot(tr_losses2, label="NN", lw=3)
    plt.legend()
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.grid(True)
    plt.title("Pérdida durante el Entrenamiento del Autoencoder")
    plt.savefig("./outputs/plots/font_autoencoder_training_loss.png")
    plt.show()

    #Graficar todas las reconstrucciones
    preds = np.round(tr.network.forward(X, training=False)).astype(np.int32)
    pixel_errors = np.abs(X - preds).sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(list(dst.keys()),pixel_errors)
    plt.ylabel("Pixels incorrectos")
    plt.grid(True)
    plt.savefig("./outputs/plots/font_autoencoder_pixel_errors.png")
    plt.show()






if __name__ == "__main__":
    from experiments.font_encoder.main import main

    main()