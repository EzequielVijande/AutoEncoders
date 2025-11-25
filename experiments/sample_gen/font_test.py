import sys
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from skimage.transform import downscale_local_mean, rescale
from keras.datasets import mnist
from scipy.stats import norm
# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from neural_network.utils.data_utils import parse_font_file, plot_latent_space
from neural_network.core.network import NeuralNetwork
from neural_network.core.auto_encoder import AutoEncoder
from neural_network.core.vae import VAE
from neural_network.core.trainer import Trainer
from neural_network.core.losses.functions import mse, mae, bce_logits
from neural_network.config import OptimizerConfig

DATASET_PATH = "resources/datasets/font.h"

def visualize_character(data, title="Character", ax=None):
    """Visualiza un carácter 5x7 como imagen."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(2, 3))
    img = data.reshape(7, 5)
    ax.imshow(img, cmap='binary', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis('off')
    return ax


def plot_reconstructions(X, X_reconstructed, char_labels, num_samples=8, save_path=None):
    """Muestra comparación lado a lado de originales y reconstruidos."""
    num_samples = min(num_samples, len(X))
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 5))

    for i in range(num_samples):
        # Original
        visualize_character(X[i], f"Original: {char_labels[i]}", axes[0, i])

        # Reconstruido
        reconstructed_binary = np.round(X_reconstructed[i])
        visualize_character(reconstructed_binary, f"Recons.: {char_labels[i]}", axes[1, i])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def evaluate_reconstruction_quality(X, X_reconstructed, char_labels):
    X_reconstructed_binary = np.round(X_reconstructed)
    pixel_errors = np.abs(X - X_reconstructed_binary).sum(axis=1)
    perfect_reconstructions = np.sum(pixel_errors == 0)
    max_1_pixel_error = np.sum(pixel_errors <= 1)
    max_2_pixel_error = np.sum(pixel_errors <= 2)
    print(f"\nTotal de patrones: {len(X)}")
    print(f"Reconstrucciones perfectas (0 errores): {perfect_reconstructions} ({100*perfect_reconstructions/len(X):.1f}%)")
    print(f"Con máximo 1 píxel de error: {max_1_pixel_error} ({100*max_1_pixel_error/len(X):.1f}%)")
    print(f"\nError promedio por patrón: {pixel_errors.mean():.2f} píxeles")
    print(f"Error máximo: {pixel_errors.max():.0f} píxeles")
    print(f"Desviación estándar del error: {pixel_errors.std():.2f}")
    return pixel_errors

def generate_font(f1, f2, vae, n_fonts=10):
    if f1[0]< f2[0]:
        pk1 = f1
        pk2 = f2
    else:
        pk1 = f2
        pk2 = f1
    diff = pk2 - pk1
    offset = np.linspace(0.0, 1.0, num=n_fonts)
    new_mus = np.zeros((n_fonts, f1.size))
    for i, off in enumerate(offset):
        new_mus[i]= pk1 + off*diff
    return vae.decode(new_mus)

DATASET_PATH = "resources/datasets/font.h"
MODEL_NAME = "VAE" #Should be 'AE' or 'VAE'

def main():
    #Cargar  parsear dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), -1))
    x_test = x_test.reshape((len(x_test), -1))
    X = x_train
    #Inicializar modelo
    topology = [784, 256, 2]
    dec_act = ['relu', 'linear']

    if MODEL_NAME == "AE":
        enc_act = ['relu', 'linear']
        model = AutoEncoder(topology, enc_act, dec_act) #Autoencoder
        kl_reg = 0
    elif MODEL_NAME == "VAE":
        enc_act = ['relu'] #VAE
        model = VAE(topology, enc_act, dec_act) #VAE
        kl_reg = 1.0
    else:
        raise ValueError('Invalid MODEL_NAME, valid arguments are \'AE\' or  \'VAE\'')

    # #Parametros de entrenamiento
    # b_size = 100 #Conjunto completo como batch
    # lr = 1e-3
    # epochs = 50
    # opt_cfg = OptimizerConfig("Adam")
    # print(f"{MODEL_NAME} Topology: {topology}")
    # print(f'Batch size = {b_size}')
    # print(f'LR = {lr}')
    # print('KL regularization weight = ', kl_reg)
    # tr = Trainer(lr, epochs, model, bce_logits, opt_cfg, kl_reg=kl_reg)
    # #Entrenar modelo
    # tr_losses, _ = tr.train(X, X, b_size, verbose=True)
    # #params
    # if MODEL_NAME == 'VAE':
    #     mu_arr, sigma_arr = model.get_params(X)
    # else:
    #     mu_arr = model.encode(X)

    # # Plot mu parameters distribution
    # plt.figure()
    # sns.kdeplot(x=mu_arr[:,0].flatten(), label="Mu 1")
    # sns.kdeplot(x=mu_arr[:,1].flatten(), label="Mu 2")
    # plt.legend()
    # plt.title("Distribución de parámetros mu")
    # plt.savefig(f"./outputs/plots/font_{MODEL_NAME}_mu_distribution.png")
    # plt.show()
    # #Plot sigma parameters distribution
    # if MODEL_NAME == 'VAE':
    #     plt.figure()
    #     sns.kdeplot(x=np.exp(sigma_arr[:,0].flatten()/2.0), label="Sigma 1")
    #     sns.kdeplot(x=np.exp(sigma_arr[:,1].flatten()/2.0), label="Sigma 2")
    #     plt.legend()
    #     plt.title("Distribución de parámetros sigma")
    #     plt.savefig("./outputs/plots/font_vae_sigma_distribution.png")
    #     plt.show()
    # model.save_weights(f"./outputs/weights/font_{MODEL_NAME}_weights.npz")

    # #Graficar perdida
    # plt.figure()
    # plt.plot(tr_losses, label=MODEL_NAME, lw=3)
    # plt.legend()
    # plt.xlabel("Épocas")
    # plt.ylabel("Pérdida")
    # plt.grid(True)
    # plt.title("Pérdida durante el Entrenamiento del Autoencoder")
    # plt.savefig(f"./outputs/plots/font_{MODEL_NAME}_training_loss.png")
    # plt.show()

    #Load weights
    model.load_weights(f"./outputs/weights/font_{MODEL_NAME}_weights.npz")
    if MODEL_NAME=='VAE':
        mu_arr, sigma_arr = model.get_params(X)
    elif MODEL_NAME=='AE':
        mu_arr = model.encode(X)
    else:
        raise ValueError('Invalid MODEL_NAME, valid arguments are \'AE\' or  \'VAE\'')

    #Plot latent space
    plot_latent_space(mu_arr[:20], y_train[:20], save_path=f"./outputs/plots/font_{MODEL_NAME}_latent_space.png")

    # Plot mu parameters distribution
    plt.figure()
    sns.kdeplot(x=mu_arr[:,0].flatten(), label="Mu 1")
    sns.kdeplot(x=mu_arr[:,1].flatten(), label="Mu 2")
    plt.legend()
    plt.title("Distribución de parámetros mu")
    plt.savefig(f"./outputs/plots/font_{MODEL_NAME}_mu_distribution.png")
    plt.show()
    #Plot sigma parameters distribution
    if MODEL_NAME == 'VAE':
        plt.figure()
        sns.kdeplot(x=np.exp(sigma_arr[:,0].flatten()/2.0), label="Sigma 1")
        sns.kdeplot(x=np.exp(sigma_arr[:,1].flatten()/2.0), label="Sigma 2")
        plt.legend()
        plt.title("Distribución de parámetros sigma")
        plt.savefig("./outputs/plots/font_vae_sigma_distribution.png")
        plt.show()

    #Graficar todas las reconstrucciones
    preds = 1 / (1+np.exp(-model.decode(mu_arr, training=False)))
    preds = np.round(preds).astype(np.int32)
    pixel_errors = np.abs(X - preds).sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(y_train[:20], pixel_errors[:20])
    plt.xticks(rotation=45)
    plt.ylabel("Pixels incorrectos")
    plt.grid(True)
    plt.savefig(f"./outputs/plots/font_{MODEL_NAME}_pixel_errors.png")
    plt.show()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(mu_arr[:,0], mu_arr[:,1], c=y_train, cmap='viridis')
    plt.colorbar()
    plt.show()

    # #Generate pokemons along a direction
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = 1 / (1+np.exp(-model.decode(z_sample, training=False)))
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()





if __name__ == "__main__":
    main()