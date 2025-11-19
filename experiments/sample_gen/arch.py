import sys
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from skimage.transform import downscale_local_mean
# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from neural_network.utils.data_utils import parse_font_file, plot_latent_space
from neural_network.core.network import NeuralNetwork
from neural_network.core.vae import VAE
from neural_network.core.trainer import Trainer
from neural_network.core.losses.functions import mse, mae
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
OUTPUT_PATH = './outputs/arch_results.csv'
LATENT_DIM = 2

def main():
    #Cargar dataset de pokemones
    pokemons2use = ['bulbasaur', 'charmander', 'squirtle', 'pikachu', 'jigglypuff',
                    'meowth', 'psyduck', 'snorlax', 'magikarp', 'eevee', 'beedrill',
                    'mewtwo', 'dragonite', 'gengar', 'lapras', 'vaporeon',
                    'flareon', 'jolteon', 'alakazam', 'machamp', 'golem', 'onix',
                    'scyther', 'magmar', 'electabuzz', 'pinsir', 'aerodactyl']
    downsample_factor = 2
    X = load_poke_dst(POKEMON_LBLS_PATH, POKEMON_DATASET_PATH, pokemons2use, downsample_factor)
    #visualize_pokemons(X, pokemons2use)

    #Parametros fijos x arquitectura
    n_hidden_layers = 4
    b_size = len(X) #Conjunto completo como batch
    lrs = np.logspace(-4, -2, num=4)
    epochs= 5000
    kl_reg = 7e0
    opt_cfg = OptimizerConfig("ADAM")
    #topologies
    topologies = [[X.shape[-1], 256, 128, 2],
                  [X.shape[-1], 256, 128, 64, 2],
                  [X.shape[-1], 256, 64, 2],
                  [X.shape[-1], 256, 2]]
    #Results to store
    results_dict = {'topology':[], 'LR':[], 'iter':[], 'loss':[],
                    'epoch':[]}
    for idx, topology in enumerate(topologies):
        print(f'Progress = {100*(idx/len(topologies))}%')
        enc_act = ['tanh']*(len(topology)-2)
        dec_act = ['tanh']*(len(topology)-2) + ['sigmoid']
        for lr in lrs:
            for i in range(5):
                vae = VAE(topology, enc_act, dec_act)
                tr = Trainer(lr, epochs, vae, mse, opt_cfg, kl_reg=kl_reg)
                #Entrenar modelo
                tr_losses, _ = tr.train(X, X, b_size)
                for j, loss in enumerate(tr_losses):
                    results_dict['topology'].append(topology)
                    results_dict['LR'].append(lr)
                    results_dict['iter'].append(i)
                    results_dict['loss'].append(loss)
                    results_dict['epoch'].append(j)
                del vae
        df = pd.DataFrame(results_dict)
        df.to_csv(OUTPUT_PATH)





if __name__ == "__main__":
    main()