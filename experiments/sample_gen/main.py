import sys
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from skimage.transform import downscale_local_mean, rescale
# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from neural_network.utils.data_utils import parse_font_file, plot_latent_space
from neural_network.core.network import NeuralNetwork
from neural_network.core.auto_encoder import AutoEncoder
from neural_network.core.vae import VAE
from neural_network.core.trainer import Trainer
from neural_network.core.losses.functions import mse, mae, bce_logits
from neural_network.config import OptimizerConfig

def load_poke_dst(lbls_path, data_path, pokemons2use=None, downsample_factor: int = 1):
    #Load labels
    poke_lbls = []
    result_ds = []
    for line in open(POKEMON_LBLS_PATH, 'r'):
        poke_lbls.append(line.split('.')[0])
    #Load pokemon dataset
    poke_ds = np.load(POKEMON_DATASET_PATH, 'r')
    if pokemons2use is not None:
        for name in pokemons2use:
            if name not in poke_lbls:
                raise ValueError(f"El Pokémon '{name}' no se encuentra en las etiquetas.")
            else:
                idx = poke_lbls.index(name)
                ds_img = downscale_local_mean(poke_ds[idx], (downsample_factor, downsample_factor)).flatten()
                ds_img[ds_img<128.0] = 0.0
                result_ds.append(ds_img.clip(max=1.0))
    else:
        for img in poke_ds:
            ds_img = downscale_local_mean(img, (downsample_factor, downsample_factor)).flatten()
            ds_img[ds_img<128.0] = 0.0
            result_ds.append(ds_img.clip(max=1.0))
    return np.array(result_ds)

def visualize_pokemons(pokemons, pokemon_names, out_path = None):
    n_pokemons = len(pokemons)
    n_cols =9
    n_rows = (n_pokemons + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    for i, ax in enumerate(axes.flat):
        if i < n_pokemons:
            ax.imshow(pokemons[i].reshape(24, 24), cmap='gray')
            ax.set_title(pokemon_names[i])
        ax.axis('off')
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
    plt.show()

def generate_pokemon(pkmn1, pkmn2, vae, n_poke=10):
    print(pkmn1.shape)
    print(pkmn2.shape)
    if pkmn1[0]< pkmn2[0]:
        pk1 = pkmn1
        pk2 = pkmn2
    else:
        pk1 = pkmn2
        pk2 = pkmn1
    diff = pk2 - pk1
    offset = np.linspace(0.0, 1.0, num=n_poke)
    new_mus = np.zeros((n_poke, pkmn1.size))
    for i, off in enumerate(offset):
        new_mus[i]= pk1 + off*diff
    return vae.decode(new_mus)

DATASET_PATH = "resources/datasets/font.h"
POKEMON_DATASET_PATH = "resources/datasets/pokesprites_pixels.npy"
POKEMON_LBLS_PATH = "resources/datasets/poke_lbls.txt"
MODEL_NAME = "VAE" #Should be 'AE' or 'VAE'

def main():
    # #Cargar dataset de pokemones
    pokemons2use = ['bulbasaur', 'charmander', 'squirtle', 'pikachu', 'jigglypuff',
                    'meowth', 'psyduck', 'snorlax', 'magikarp', 'eevee', 'beedrill',
                    'mewtwo', 'dragonite', 'gengar', 'lapras', 'vaporeon',
                    'flareon', 'jolteon', 'alakazam', 'machamp', 'golem', 'onix',
                    'scyther', 'magmar', 'electabuzz', 'pinsir', 'aerodactyl']
    downsample_factor = 2
    X = load_poke_dst(POKEMON_LBLS_PATH, POKEMON_DATASET_PATH, pokemons2use, downsample_factor)
    # visualize_pokemons(X, pokemons2use)

    #Inicializar modelo
    topology = [576, 64, 2]
    dec_act = ['tanh', 'linear']

    if MODEL_NAME == "AE":
        enc_act = ['tanh','tanh'] #AE
        model = AutoEncoder(topology, enc_act, dec_act) #Autoencoder
        kl_reg = 0
    elif MODEL_NAME == "VAE":
        enc_act = ['tanh'] #VAE
        model = VAE(topology, enc_act, dec_act) #VAE
        kl_reg = 1e1
    else:
        raise ValueError('Invalid MODEL_NAME, valid arguments are \'AE\' or  \'VAE\'')

    # #Parametros de entrenamiento
    # b_size = len(X) #Conjunto completo como batch
    # lr = 1e-3
    # epochs = 20000
    # opt_cfg = OptimizerConfig("ADAM")
    # print(f"{MODEL_NAME} Topology: {topology}")
    # print(f'Batch size = {b_size}')
    # print(f'LR = {lr}')
    # print('KL regularization weight = ', kl_reg)
    # tr = Trainer(lr, epochs, model, bce_logits, opt_cfg, kl_reg=kl_reg)
    # #Entrenar modelo
    # tr_losses, _ = tr.train(X, X, b_size)
    # #params
    # if MODEL_NAME == 'VAE':
    #     mu_arr, sigma_arr = model.get_params(X)
    # else:
    #     mu_arr = model.encode(X)

    # model.save_weights(f"./outputs/weights/poke_{MODEL_NAME}_weights.npz")

    # #Graficar perdida
    # plt.figure()
    # plt.plot(tr_losses, label=MODEL_NAME, lw=3)
    # plt.legend()
    # plt.xlabel("Épocas")
    # plt.ylabel("Pérdida")
    # plt.yscale("log")
    # plt.grid(True)
    # plt.title("Pérdida durante el Entrenamiento del Autoencoder")
    # plt.savefig(f"./outputs/plots/poke_{MODEL_NAME}_training_loss.png")
    # plt.show()

    # #Load weights
    model.load_weights(f"./outputs/weights/poke_{MODEL_NAME}_weights.npz")
    if MODEL_NAME=='VAE':
        mu_arr, sigma_arr = model.get_params(X)
    elif MODEL_NAME=='AE':
        mu_arr = model.encode(X)
    else:
        raise ValueError('Invalid MODEL_NAME, valid arguments are \'AE\' or  \'VAE\'')

  # Plot mu parameters distribution
    plt.figure()
    sns.kdeplot(x=mu_arr[:,0].flatten(), label="Mu 1")
    sns.kdeplot(x=mu_arr[:,1].flatten(), label="Mu 2")
    plt.legend()
    plt.title("Distribución de parámetros mu")
    plt.savefig(f"./outputs/plots/poke_{MODEL_NAME}_mu_distribution.png")
    plt.show()
    #Plot sigma parameters distribution
    if MODEL_NAME == 'VAE':
        plt.figure()
        sns.kdeplot(x=np.exp(sigma_arr[:,0].flatten()/2.0), label="Sigma 1")
        sns.kdeplot(x=np.exp(sigma_arr[:,1].flatten()/2.0), label="Sigma 2")
        plt.legend()
        plt.title("Distribución de parámetros sigma")
        plt.savefig("./outputs/plots/poke_vae_sigma_distribution.png")
        plt.show()

    #Plot latent space
    plot_latent_space(mu_arr, pokemons2use, save_path=f"./outputs/plots/poke_{MODEL_NAME}_latent_space.png")

    #Graficar todas las reconstrucciones
    preds = 1 / (1+np.exp(-model.decode(mu_arr, training=False)))
    preds = np.round(preds).astype(np.int32)
    # preds = np.round(model.decode(mu_arr, training=False)).astype(np.int32)
    pixel_errors = np.abs(X - preds).sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(pokemons2use, pixel_errors)
    plt.xticks(rotation=45)
    plt.ylabel("Pixels incorrectos")
    plt.grid(True)
    plt.savefig(f"./outputs/plots/poke_{MODEL_NAME}_pixel_errors.png")
    plt.show()


    # #Generate pokemons along a direction
    n_poke = 18
    pk1 = pokemons2use.index('lapras')
    pk2 = pokemons2use.index('beedrill')
    new_gen = generate_pokemon(mu_arr[pk1], mu_arr[pk2], model, n_poke)
    names = [f'p{i}' for i in range(n_poke)]
    new_gen = 1 / (1+np.exp(-new_gen))
    # for gen in new_gen:
    #     img = gen.reshape((24,24))
    #     upsampld = rescale(img, 2, order=3)
    #     plt.figure()
    #     plt.imshow(upsampld,cmap='gray')
    #     plt.show()
    visualize_pokemons(new_gen, names, f'outputs/plots/{MODEL_NAME}_gens.png')





if __name__ == "__main__":
    main()