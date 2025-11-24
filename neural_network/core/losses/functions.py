# loss_functions.py

import numpy as np
from typing import Tuple

# -------------------------------
# Función de Pérdida: Softmax con Entropía Cruzada
# -------------------------------
def softmax_cross_entropy_with_logits(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    logits_stable = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    epsilon = 1e-8
    softmax_probs = np.clip(softmax_probs, epsilon, 1. - epsilon)
    loss = -np.sum(labels * np.log(softmax_probs)) / logits.shape[0]
    return loss, softmax_probs-labels


def mae(x1, x2) -> Tuple[float, np.ndarray]:
    "Mean absolute err"
    return np.mean(np.abs(x1-x2)), np.sign(x1-x2)/x1.shape[0]

def mse(pred, label) -> Tuple[float, np.ndarray]:
    "Mean absolute err"
    return np.mean((pred-label)**2), pred-label

def bce(y: np.ndarray, t: np.ndarray, eps: float = 1e-12) -> Tuple[float, np.ndarray]:
    """ Binary cross-entropy """
    y_clipped = np.clip(y, eps, 1.0 - eps)
    loss = -np.mean(t * np.log(y_clipped) + (1 - t) * np.log(1 - y_clipped))
    grad = (y_clipped - t) / (y_clipped * (1.0 - y_clipped))
    return loss, grad


def kl_loss(mean, var) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    "KL divergence"
    loss = - 0.5 * np.sum(1 + var - (mean**2) - np.exp(var), axis=-1)
    return  (loss.mean(), ( mean, -0.5*(1-np.exp(var))) )
