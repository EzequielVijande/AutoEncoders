import sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from neural_network.utils.data_utils import parse_font_file
from neural_network.core.network import NeuralNetwork
from neural_network.core.vae import VAE
from neural_network.core.trainer import Trainer
from neural_network.core.losses.functions import mse, mae
from neural_network.config import OptimizerConfig

DATASET_PATH = "resources/datasets/font.h"


def main():
    #Cargar  parsear dataset
    dst = parse_font_file(DATASET_PATH)
    X = np.zeros((len(dst), len(dst[list(dst.keys())[0]])))
    for i, key in enumerate(dst):
        X[i] = dst[key]
    #Inicializar modelo AE
    topology = [35, 26, 16, 8, 2]
    enc_act = ['relu']*len(topology[2:])
    dec_act = ['relu'] *len(topology[2:]) + ['sigmoid']
    vae = VAE(topology, enc_act, dec_act)
    #Parametros de entrenamiento
    b_size = 32 #Conjunto completo como batch
    lr = 1e-2
    epochs = 10000
    opt_cfg = OptimizerConfig("ADAM")
    tr = Trainer(lr, epochs, vae, mse, opt_cfg)
    #Entrenar modelo
    tr_losses, _ = tr.train(X, X, b_size)
    #Verificar diferencias entre codificado-decodificado y forward
    encoded = vae.decode(vae.encode(X, training=False), training=False)
    encoded2 = vae.forward(X, training=False)
    print(f'difference between encoded-decoded and forward: {np.sum(np.abs(encoded - encoded2))}')
    #params
    print(vae.get_params(X[0]))
    #Graficar perdida
    plt.figure()
    plt.plot(tr_losses, label="VAE", lw=3)
    plt.legend()
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.grid(True)
    plt.title("Pérdida durante el Entrenamiento del Autoencoder")
    plt.savefig("./outputs/plots/vae_training_loss.png")
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
    main()