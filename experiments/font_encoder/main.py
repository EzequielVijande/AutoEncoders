import sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import argparse
import yaml
import json
# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from neural_network.utils.data_utils import parse_font_file, plot_latent_space
from neural_network.core.auto_encoder import AutoEncoder
from neural_network.core.trainer import Trainer
from neural_network.core.losses import functions as loss_funcs
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


def plot_reconstructions(X, X_reconstructed, char_labels, num_samples=8, show_plots=True, save_path=None):
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
    if show_plots: plt.show()

def plot_denoising_examples(X, X_noisy, X_recons, labels, num_samples=8, save_path=None):
    """
    Original / Noisy / Denoised for some characters.
    """
    num_samples = min(num_samples, len(X))
    fig, axes = plt.subplots(3, num_samples, figsize=(2 * num_samples, 7))

    for i in range(num_samples):
        # File 0: original
        visualize_character(X[i], f"Original {labels[i]}", axes[0, i])
        # File 1: with noise
        visualize_character(X_noisy[i], "Noisy", axes[1, i])
        # Fila 2: reconstructed (denoised)
        visualize_character(np.round(X_recons[i]), "Denoised", axes[2, i])

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

def add_binary_noise(X, flip_prob=0.2):
    """
    Adds binary noise (flips 0↔1) with probability flip_prob for each pixel.
    X: array shaped (n_samples, 35) with values 0/1.
    """
    X_noisy = X.copy()
    mask = np.random.rand(*X.shape) < flip_prob
    X_noisy[mask] = 1 - X_noisy[mask]  # 0 -> 1, 1 -> 0
    return X_noisy

def eval_unseen_noise(dae, X, p, n_trials=20):
    """Evaluate denoising performance on many new random noise samples."""
    errs = []
    for _ in range(n_trials):
        X_unseen_noise = add_binary_noise(X, flip_prob=p)
        X_unseen_recons = dae.forward(X_unseen_noise, training=False)
        pixel_errors_unseen = np.abs(np.round(X_unseen_recons) - X).sum(axis=1)
        errs.append(pixel_errors_unseen.mean())
    return float(np.mean(errs)), float(np.std(errs))

def train_denoising_autoencoder(X, flip_levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6), epochs=5000):
    """
    trains a denoising autoencoder for different levels of noise.
    Returns a dict {p: {"model": dae, "losses": [...], "pixel_errors": [...]}}.
    """
    results = {}

    for p in flip_levels:
        print(f"\n==== Training Denoising AE with noise level p={p} ====")

        # 1) Noisy entry
        X_noisy = add_binary_noise(X, flip_prob=p)

        # 2) Same basic architecture as AE
        topology = [35, 24, 12, 2]
        enc_act = ['tanh', 'tanh', 'linear']
        dec_act = ['tanh', 'tanh', 'sigmoid']
        dae = AutoEncoder(topology, enc_act, dec_act)

        # 3) Training: input = X_noisy, target = X limpio
        lr = 1e-3
        b_size = 32
        opt_cfg = OptimizerConfig("ADAM")
        tr = Trainer(lr, epochs, dae, loss_funcs.bce, opt_cfg)

        losses, _ = tr.train(X_noisy, X, b_size)
        X_recons = dae.forward(X_noisy, training=False)

        # 4) Error en píxeles vs X limpio (ruido de entrenamiento)
        pixel_errors = np.abs(np.round(X_recons) - X).sum(axis=1)
        print(f"Avg pixel error (training noise p={p}): {np.mean(pixel_errors):.2f}")

        # 5) Error en ruido NO visto (promediado sobre varios ensayos)
        mean_unseen, std_unseen = eval_unseen_noise(dae, X, p, n_trials=20)
        print(f"Avg pixel error on *unseen* noise (p={p}): {mean_unseen:.2f} ± {std_unseen:.2f}")

        results[p] = {
            "model": dae,
            "losses": losses,
            "pixel_errors": pixel_errors,  # training noise
            "unseen_mean": mean_unseen,  # averaged unseen noise
            "unseen_std": std_unseen,
            "X_noisy": X_noisy,
            "X_recons": X_recons
        }

    return results


def plot_noise_vs_error(results, save_path=None):
    """
    Graphs noise level (p) vs median error in pixels using *unseen* noise.
    """
    ps = []
    means = []
    stds = []

    for p, res in sorted(results.items()):
        ps.append(p)
        means.append(res.get("unseen_mean", 0.0))
        stds.append(res.get("unseen_std", 0.0))

    plt.figure(figsize=(8, 5))
    plt.errorbar(ps, means, yerr=stds, fmt='-o', capsize=5, linewidth=2)
    plt.title("Efecto del nivel de ruido en la reconstrucción (ruido no visto)")
    plt.xlabel("Probabilidad de ruido (p)")
    plt.ylabel("Error promedio en píxeles")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def load_config(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p, 'r') as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and 'experiments' in data:
        return data['experiments']
    return data


def save_results_json(output_dir, name, results):
    arch_logs = Path(output_dir) / 'arch_logs'
    arch_logs.mkdir(parents=True, exist_ok=True)
    out_path = arch_logs / f"{name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    return out_path


def generate_interpolated_char(char1_key, char2_key, ae, dst, num_steps=5, show_plots=True, save_path=None):
    """Genera caracteres interpolando entre dos caracteres en el espacio latente."""
    char1_data = dst[char1_key]
    char2_data = dst[char2_key]
    latent1 = ae.encode(np.array([char1_data]), training=False)[0]
    latent2 = ae.encode(np.array([char2_data]), training=False)[0]
    interpolated_latents = np.array([latent1 + (latent2 - latent1) * t for t in np.linspace(0, 1, num_steps)])
    interpolated_images = ae.decode(interpolated_latents, training=False)

    fig, axes = plt.subplots(1, num_steps + 2, figsize=(2 * (num_steps + 2), 3))
    visualize_character(char1_data, f"Original: {char1_key}", ax=axes[0])
    for i, img_data in enumerate(interpolated_images):
        reconstructed_binary = np.round(img_data)
        visualize_character(reconstructed_binary, f"Interp. {i+1}", ax=axes[i+1])
    visualize_character(char2_data, f"Original: {char2_key}", ax=axes[num_steps + 1])
    plt.tight_layout()
    if save_path:   plt.savefig(save_path, dpi=150)
    if show_plots:  plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to yaml config for this run')
    args = parser.parse_args()
    cfg = None
    if args.config is not None:
        cfg = load_config(args.config)

    # Default params
    name = 'font_ae'
    topology = [35, 24, 12, 2]
    enc_act = ['tanh', 'tanh', 'linear']
    dec_act = ['tanh', 'tanh', 'sigmoid']
    b_size = 32
    lr = 1e-3
    epochs = 10000
    optimizer = 'ADAM'
    output_dir = './outputs'
    show_plots = True
    loss = 'bce'

    if cfg is not None:
        name = cfg.get('name', name)
        topology = cfg.get('topology', topology)
        enc_act = cfg.get('enc_act', enc_act)
        dec_act = cfg.get('dec_act', dec_act)
        b_size = cfg.get('b_size', b_size)
        lr = cfg.get('lr', lr)
        epochs = cfg.get('epochs', epochs)
        optimizer = cfg.get('optimizer', optimizer)
        output_dir = cfg.get('output_dir', output_dir)
        show_plots = cfg.get('show_plots', show_plots)
        loss = cfg.get('loss', loss)

    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    #Cargar  parsear dataset
    dst = parse_font_file(DATASET_PATH)
    X = np.zeros((len(dst), len(dst[list(dst.keys())[0]])))

    for i, key in enumerate(dst):
        X[i] = dst[key]
    #Inicializar modelo AE
    ae = AutoEncoder(topology, enc_act, dec_act)
    loss_func = getattr(loss_funcs, loss)
    opt_cfg = OptimizerConfig(optimizer)
    tr = Trainer(lr, epochs, ae, loss_func, opt_cfg)
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
    plt.plot(tr_losses, label=f"AE: {topology}", lw=3, alpha=0.9, color='purple')
    plt.legend(fontsize=14)
    plt.xlabel("Épocas", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.title("Pérdida durante el Entrenamiento", fontsize=15, fontweight='bold')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(str(plots_dir / "font_autoencoder_training_loss.png"))
    if show_plots: plt.show()

    #Graficar todas las reconstrucciones
    plt.figure(figsize=(14, 6))
    colors = ['green' if e <= 1 else 'orange' if e <= 2 else 'red' for e in pixel_errors]
    bars = plt.bar(char_labels, pixel_errors, color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(y=1, color='green', linestyle='--', label='Objetivo: ≤1 píxel', linewidth=2)
    plt.ylabel("Pixels incorrectos", fontsize=20)
    plt.xlabel("Carácter", fontsize=20)
    plt.xticks(rotation=0, ha='right', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(str(plots_dir / "font_autoencoder_pixel_errors.png"), dpi=150)
    if show_plots: plt.show()

    # Espacio latente 2D
    latent = ae.encode(X, training=False)
    plot_latent_space(latent, char_labels,
                     save_path=str(plots_dir / "font_autoencoder_latent_space.png"))
    # Comparación de reconstrucciones (todos los caracteres)
    plot_reconstructions(X, x_reconstructed, char_labels, num_samples=len(X), show_plots=show_plots,
                        save_path=str(plots_dir / "font_autoencoder_all_reconstructions.png"))
    # Comparación detallada de los peores casos
    worst_indices = np.argsort(pixel_errors)[-8:][::-1]
    if len(worst_indices) > 0:
        X_worst = X[worst_indices]
        X_worst_recons = x_reconstructed[worst_indices]
        worst_labels = [char_labels[i] for i in worst_indices]
        plot_reconstructions(X_worst, X_worst_recons, worst_labels,
                           num_samples=len(worst_indices),
                           show_plots=show_plots,
                           save_path=str(plots_dir / "font_autoencoder_worst_cases.png"))

    # Generar caracteres interpolados
    generate_interpolated_char('p', 'r', ae, dst, num_steps=5, show_plots=show_plots,
                               save_path=str(plots_dir / "font_autoencoder_interpolated.png"))

    results = {
        'name': name,
        'topology': topology,
        'final_loss': float(tr_losses[-1]) if len(tr_losses) > 0 else None,
        'loss_history': [float(loss) for loss in tr_losses],
        'pixel_errors_mean': float(pixel_errors.mean()),
        'pixel_errors_std': float(pixel_errors.std())
    }
    out_path = save_results_json(output_dir, name, results)
    print(f"Saved results to {out_path}")


    #  Denoising Autoencoder (1.b)
    print("\n=== Denoising Autoencoder experiments ===")
    noise_levels = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    denoise_results = train_denoising_autoencoder(X, flip_levels=noise_levels, epochs=3000)

    # User noise level for visual example (eg. p=0.2)
    p_example = 0.2
    if p_example in denoise_results:
        res_p = denoise_results[p_example]
        X_noisy_p = res_p["X_noisy"]
        X_recons_p = res_p["X_recons"]
        plot_denoising_examples(
            X,
            X_noisy_p,
            X_recons_p,
            char_labels,
            num_samples=len(X),
            save_path="./outputs/plots/font_denoising_examples_p02.png"
        )

    # curve noise vs median noise
    plot_noise_vs_error(
        denoise_results,
        save_path="./outputs/plots/font_denoising_noise_vs_error.png"
    )

if __name__ == "__main__":
    main()