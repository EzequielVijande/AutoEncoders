import sys
from pathlib import Path
from matplotlib import pyplot as plt
# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from neural_network.utils.data_utils import parse_font_file
from neural_network.core.network import NeuralNetwork
from neural_network.core.trainer import Trainer

DATASET_PATH = "resources/datasets/font.h"


def main():
    #Cargar  parsear dataset
    dst = parse_font_file(DATASET_PATH)
    for key in dst:
        font = dst[key].reshape(7,-1)
        plt.figure()
        plt.imshow(font, cmap='binary')
        plt.show()
    topology = [35, 16, 8, 2, 8, 16, 35] # Simetría en el autoencoder






if __name__ == "__main__":
    from experiments.font_encoder.main import main

    main()