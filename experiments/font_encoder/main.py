import sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from neural_network.utils.data_utils import parse_font_file, plot_latent_space
from neural_network.core.auto_encoder import AutoEncoder
from neural_network.core.trainer import Trainer
from neural_network.core.losses.functions import mse
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

def main():
    #Cargar  parsear dataset
    dst = parse_font_file(DATASET_PATH)
    X = np.zeros((len(dst), len(dst[list(dst.keys())[0]])))

    for i, key in enumerate(dst):
        X[i] = dst[key]
    #Inicializar modelo AE
    topology = [35, 24, 12, 2]
    enc_act = ['tanh', 'tanh', 'linear']
    dec_act = ['tanh', 'tanh', 'sigmoid']
    ae = AutoEncoder(topology, enc_act, dec_act)
    #Parametros de entrenamiento
    b_size = 32 #Conjunto completo como batch
    lr = 1e-3
    epochs = 10000
    opt_cfg = OptimizerConfig("ADAM")
    tr = Trainer(lr, epochs, ae, mse, opt_cfg)
    #Entrenar modelo
    tr_losses, _ = tr.train(X, X, b_size)
    #Verificar diferencias entre codificado-decodificado y forward
    encoded = ae.decode(ae.encode(X, training=False), training=False)
    x_reconstructed = ae.forward(X, training=False)
    print(f"Loss final: {tr_losses[-1]:.6f}")
    print(f'difference between encoded-decoded and forward: {np.sum(np.abs(encoded - x_reconstructed))}')
    char_labels = list(dst.keys())
    pixel_errors = evaluate_reconstruction_quality(X, x_reconstructed, char_labels)
    #Graficar perdida
    plt.figure(figsize=(14, 7))
    plt.plot(tr_losses, label="AE: [35, 24, 12, 2]", lw=3, alpha=0.9, color='purple')
    plt.legend(fontsize=14)
    plt.xlabel("Épocas", fontsize=14)
    plt.ylabel("Pérdida (MSE)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.title("Pérdida durante el Entrenamiento", fontsize=15, fontweight='bold')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./outputs/plots/font_autoencoder_training_loss.png")
    plt.show()

    #Graficar todas las reconstrucciones
    plt.figure(figsize=(14, 6))
    colors = ['green' if e <= 1 else 'orange' if e <= 2 else 'red' for e in pixel_errors]
    bars = plt.bar(char_labels, pixel_errors, color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(y=1, color='green', linestyle='--', label='Objetivo: ≤1 píxel', linewidth=2)
    plt.ylabel("Pixels incorrectos", fontsize=14)
    plt.xlabel("Carácter", fontsize=14)
    plt.xticks(rotation=0, ha='right')
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("./outputs/plots/font_autoencoder_pixel_errors.png", dpi=150)
    plt.show()

    # Espacio latente 2D
    latent = ae.encode(X, training=False)
    plot_latent_space(latent, char_labels,
                     save_path="./outputs/plots/font_autoencoder_latent_space.png")
    # Comparación de reconstrucciones (todos los caracteres)
    plot_reconstructions(X, x_reconstructed, char_labels, num_samples=len(X),
                        save_path="./outputs/plots/font_autoencoder_all_reconstructions.png")
    # Comparación detallada de los peores casos
    worst_indices = np.argsort(pixel_errors)[-8:][::-1]
    if len(worst_indices) > 0:
        X_worst = X[worst_indices]
        X_worst_recons = x_reconstructed[worst_indices]
        worst_labels = [char_labels[i] for i in worst_indices]
        plot_reconstructions(X_worst, X_worst_recons, worst_labels,
                           num_samples=len(worst_indices),
                           save_path="./outputs/plots/font_autoencoder_worst_cases.png")



if __name__ == "__main__":
    main()