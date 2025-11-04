import numpy as np
from typing import List, Tuple, Dict, Any, Union



def shuffle_data(inputs: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Mezcla los datos de entrada y etiquetas de manera aleatoria.

    :param inputs: Array de entradas.
    :param labels: Array de etiquetas.
    :return: Tupla de arrays mezclados (inputs, labels).
    """
    assert len(inputs) == len(labels), "Las entradas y etiquetas deben tener la misma longitud."
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    return inputs[indices], labels[indices]

def k_fold_split(n_samples: int, k: int = 5, shuffle: bool = True, random_state: int = None) -> List[Tuple]:
    """
    Generate indices for k-fold cross-validation splits.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples in the dataset
    k : int
        Number of folds
    shuffle : bool
        Whether to shuffle the data before splitting
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    folds : List[Tuple]
        List of tuples (train_indices, test_indices) for each fold
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    # Calculate fold sizes
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    
    current = 0
    folds = []
    
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_indices, test_indices))
        current = stop
    
    return folds

def stratified_k_fold_split(y: np.ndarray, k: int = 5, shuffle: bool = True, random_state: int = None) -> List[Tuple]:
    """
    Generate indices for stratified k-fold cross-validation.
    Preserves the percentage of samples for each class.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(y)
    unique_classes = np.unique(y)
    
    # Get class counts and indices
    class_indices = {}
    for class_label in unique_classes:
        class_indices[class_label] = np.where(y == class_label)[0]
    
    # Initialize fold indices
    fold_indices = [[] for _ in range(k)]
    
    for class_label, indices in class_indices.items():
        class_indices_arr = indices.copy()
        
        if shuffle:
            np.random.shuffle(class_indices_arr)
        
        # Distribute class samples across folds
        n_class_samples = len(class_indices_arr)
        fold_sizes = np.full(k, n_class_samples // k, dtype=int)
        fold_sizes[:n_class_samples % k] += 1
        
        current = 0
        for fold_idx, fold_size in enumerate(fold_sizes):
            start, stop = current, current + fold_size
            fold_indices[fold_idx].extend(class_indices_arr[start:stop])
            current = stop
    
    # Convert to train-test splits
    folds = []
    for fold_idx in range(k):
        test_indices = np.array(fold_indices[fold_idx])
        train_indices = np.concatenate([fold_indices[i] for i in range(k) if i != fold_idx])
        folds.append((train_indices, test_indices))
    
    return folds

def mse(x1, x2):
    return np.mean((x1-x2)**2)

def mae(x1, x2):
    return np.mean(np.abs(x1-x2))

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy classification score."""
    if len(y_true) != len(y_pred):
        raise ValueError("Arrays must have the same length")
    
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)


def calculate_score(y_true: np.ndarray, y_pred: np.ndarray, scoring: str) -> float:
    """Calculate evaluation score based on specified metric."""
    scoring_functions = {
        'accuracy': accuracy_score,
        'mse': mse,
        'mae': mae
    }
    if scoring not in scoring_functions:
        raise ValueError(f"Unsupported scoring metric: {scoring}. Available: {list(scoring_functions.keys())}")
    
    return scoring_functions[scoring](y_true, y_pred)



# def save_weights(neural_network, file_path: str) -> None:
#     data = {}
#     for layer_num, layer in enumerate(neural_network.layers):
#         layer_weights = [perceptron.weights.tolist() for perceptron in layer.perceptrons]
#         layer_biases = [perceptron.bias for perceptron in layer.perceptrons]
#         data[f'layer_{layer_num}_weights'] = layer_weights
#         data[f'layer_{layer_num}_biases'] = layer_biases
#     np.savez(file_path, **data)
#     print(f"Pesos guardados en '{file_path}'.")
#
#
# def load_weights(neural_network, file_path: str) -> None:
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
#
#     data = np.load(file_path, allow_pickle=True)
#     num_layers = len(neural_network.layers)
#     for layer_num in range(num_layers):
#         layer_weights = data[f'layer_{layer_num}_weights']
#         layer_biases = data[f'layer_{layer_num}_biases']
#         for perceptron, w, b in zip(neural_network.layers[layer_num].perceptrons, layer_weights, layer_biases):
#             perceptron.weights = np.array(w)
#             perceptron.bias = float(b)
#     print(f"Pesos cargados desde '{file_path}'.")
