import sys
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from skimage.transform import downscale_local_mean
import seaborn as sns
# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from neural_network.utils.data_utils import parse_font_file, plot_latent_space
from neural_network.core.network import NeuralNetwork
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

def visualize_pokemons(pokemons, pokemon_names):
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
    plt.show()

DATASET_PATH = "resources/datasets/font.h"
POKEMON_DATASET_PATH = "resources/datasets/pokesprites_pixels.npy"
POKEMON_LBLS_PATH = "resources/datasets/poke_lbls.txt"
OUTPUT_PATH = './outputs/lr_searcher.csv'
LATENT_DIM = 2

def main():
    # #Cargar dataset de pokemones
    # pokemons2use = ['bulbasaur', 'charmander', 'squirtle', 'pikachu', 'jigglypuff',
    #                 'meowth', 'psyduck', 'snorlax', 'magikarp', 'eevee', 'beedrill',
    #                 'mewtwo', 'dragonite', 'gengar', 'lapras', 'vaporeon',
    #                 'flareon', 'jolteon', 'alakazam', 'machamp', 'golem', 'onix',
    #                 'scyther', 'magmar', 'electabuzz', 'pinsir', 'aerodactyl']
    # downsample_factor = 2
    # X = load_poke_dst(POKEMON_LBLS_PATH, POKEMON_DATASET_PATH, pokemons2use, downsample_factor)
    # #visualize_pokemons(X, pokemons2use)

    # #Parametros fijos x arquitectura
    # b_size = len(X) #Conjunto completo como batch
    # lrs = np.linspace(2e-4, 2e-3, num=10)
    # epochs= 20000
    # kl_reg = 1e1
    # opt_cfg = OptimizerConfig("ADAM")
    # #Results to store
    # results_dict = {'LR':[], 'iter':[], 'loss':[],
    #                 'epoch':[]}
    # topology = [576, 64, 2]
    # enc_act = ['tanh']
    # dec_act = ['tanh'] + ['linear']
    # for lr in lrs:
    #     for i in range(5):
    #         vae = VAE(topology, enc_act, dec_act)
    #         tr = Trainer(lr, epochs, vae, bce_logits, opt_cfg, kl_reg=kl_reg)
    #         #Entrenar modelo
    #         tr_losses, _ = tr.train(X, X, b_size)
    #         for j, loss in enumerate(tr_losses):
    #             results_dict['LR'].append(lr)
    #             results_dict['iter'].append(i)
    #             results_dict['loss'].append(loss)
    #             results_dict['epoch'].append(j)
    #         del vae
    # df = pd.DataFrame(results_dict)
    # df.to_csv(OUTPUT_PATH)

    #load and plot results
    df = pd.read_csv(OUTPUT_PATH)
    df['LR']= df['LR'].round(4)
    df = df.astype({'LR': str})

    print(plt.rcParams.keys())
    plt.figure(figsize=(12,8))
    sns.lineplot(df, x='epoch', y='loss', hue='LR', ci=None, lw=4)
    plt.title('VAE Training curves')
    plt.yscale('log')
    plt.rcParams['font.size'] = 22
    plt.rcParams["axes.titlesize"] = 26
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 18
    plt.savefig("./outputs/plots/vae_lrs.png")
    plt.show()





if __name__ == "__main__":
    main()