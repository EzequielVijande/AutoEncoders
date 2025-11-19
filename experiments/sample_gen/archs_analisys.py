import sys
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from skimage.transform import downscale_local_mean
# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

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

POKEMON_DATASET_PATH = "resources/datasets/pokesprites_pixels.npy"
POKEMON_LBLS_PATH = "resources/datasets/poke_lbls.txt"
OUTPUT_PATH = './outputs/arch_results.csv'
LATENT_DIM = 2

def main():
    df = pd.read_csv(OUTPUT_PATH)
    print(df.keys())
    lrs = df.LR.unique()
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(30, 14))
    for i, lr in enumerate(lrs):
        lr_df = df[df.LR == lr]
        # Plot on the first subplot
        sns.lineplot(data=lr_df, x='epoch', y='loss', hue='topology', ax=axes[i//3,i%3])
        axes[i//3,i%3].set_title(f'LR = {lr}')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig("./outputs/plots/vae_archs_loss.png")
    plt.show()



if __name__ == "__main__":
    main()