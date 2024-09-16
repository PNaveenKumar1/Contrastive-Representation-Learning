import torch
import numpy as np
from typing import Tuple
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

#Naveen imports start

import random

#ends

def get_data(
        data_path: str = 'data/cifar10_train.npz', is_linear: bool = False,
        is_binary: bool = False, grayscale: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Load CIFAR-10 dataset from the given path and return the images and labels.
    If is_linear is True, the images are reshaped to 1D array.
    If grayscale is True, the images are converted to grayscale.

    Args:
    - data_path: string, path to the dataset
    - is_linear: bool, whether to reshape the images to 1D array
    - is_binary: bool, whether to convert the labels to binary
    - grayscale: bool, whether to convert the images to grayscale

    Returns:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    '''
    data = np.load(data_path)
    X = data['images']
    try:
        y = data['labels']
    except KeyError:
        y = None
    X = X.transpose(0, 3, 1, 2)
    if is_binary:
        idxs0 = np.where(y == 0)[0]
        idxs1 = np.where(y == 1)[0]
        idxs = np.concatenate([idxs0, idxs1])
        X = X[idxs]
        y = y[idxs]
    if grayscale:
        X = convert_to_grayscale(X)
    if is_linear:
        X = X.reshape(X.shape[0], -1)
    
    # HINT: rescale the images for better (and more stable) learning and performance

    return X, y


def convert_to_grayscale(X: np.ndarray) -> np.ndarray:
    '''
    Convert the given images to grayscale.

    Args:
    - X: np.ndarray, images in RGB format

    Returns:
    - X: np.ndarray, grayscale images
    '''
    return np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])


def train_test_split(
        X: np.ndarray, y: np.ndarray, test_ratio: int = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Split the given dataset into training and test sets.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - test_ratio: float, ratio of the test set

    Returns:
    - X_train: np.ndarray, training images
    - y_train: np.ndarray, training labels
    - X_test: np.ndarray, test images
    - y_test: np.ndarray, test labels
    '''
    assert test_ratio < 1 and test_ratio > 0

    data_rows = X.shape[0]

    test_rows = int(data_rows * test_ratio)
    train_rows = data_rows - test_rows

    X_train = X[:train_rows]
    y_train = y[:train_rows]

    X_test = X[train_rows:]
    y_test = y[train_rows:]

    # X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio)

    # raise NotImplementedError('Split the dataset here')
    
    return X_train, y_train, X_test, y_test


def get_data_batch(
        X: np.ndarray, y: np.ndarray, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Get a batch of the given dataset.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - batch_size: int, size of the batch

    Returns:
    - X_batch: np.ndarray, batch of images
    - y_batch: np.ndarray, batch of labels
    '''

    n = X.shape[0]
    
    idxs = random.sample(range(0,n), batch_size) # TODO: get random indices of the batch size without replacement from the dataset
    return X[idxs], y[idxs]


# TODO: Read up on generator functions online
def get_contrastive_data_batch(
        X: np.ndarray, y: np.ndarray, batch_size: int
):  # Yields: Tuple[np.ndarray, np.ndarray, np.ndarray]
    '''
    Get a batch of the given dataset for contrastive learning.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - batch_size: int, size of the batch

    Yields:
    - X_a: np.ndarray, batch of anchor samples
    - X_p: np.ndarray, batch of positive samples
    - X_n: np.ndarray, batch of negative samples
    '''
    n = X.shape[0]

    ids = np.random.choice(n, batch_size, replace=False)

    while True:
        ids = np.random.choice(n, batch_size, replace=False)
        X_a = X[ids]
        labels = y[ids]
        X_p = np.zeros_like(X_a)
        X_n = np.zeros_like(X_a)
        
        for i in range(batch_size):
            label = labels[i]
            pos_idx = np.where(y == label)[0]
            neg_idx = np.where(y != label)[0]

            pos_idx = np.delete(pos_idx, np.where(pos_idx == ids[i]))
            neg_idx = np.delete(neg_idx, np.where(neg_idx == ids[i]))

            X_p[i] = X[np.random.choice(pos_idx)]
            X_n[i] = X[np.random.choice(neg_idx)]
        
        yield X_a, X_p, X_n


    # raise NotImplementedError('Get a batch of anchor, positive, and negative samples here')


def plot_losses(
        train_losses: list, val_losses: list, title: str
) -> None:
    '''
    Plot the training and validation losses.

    Args:
    - train_losses: list, training losses
    - val_losses: list, validation losses
    - title: str, title of the plot
    '''
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(f'images/loss-{title}.png')
    plt.close()


def plot_accuracies(
        train_accs: list, val_accs: list, title: str
) -> None:
    '''
    Plot the training and validation accuracies.

    Args:
    - train_accs: list, training accuracies
    - val_accs: list, validation accuracies
    - title: str, title of the plot
    '''
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.savefig(f'images/acc-{title}.png')
    plt.close()


def plot_tsne(
       z: np.ndarray, y: np.ndarray 
) -> None:
    '''
    Plot the 2D t-SNE of the given representation.

    Args:
    - z: torch.Tensor, representation
    - y: torch.Tensor, labels
    '''
 
    tsne = TSNE(n_components=2, random_state=42)
    z2 = tsne.fit_transform(z)
    y = y.detach().cpu().numpy()
    plt.figure(figsize=(20, 20))
    for i in range(10):
        idxs = np.where(y == i)
        plt.scatter(z2[idxs, 0], z2[idxs, 1], label=str(i))
        
    plt.title('t-SNE Representation')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.savefig('images/2.3.png')
    print('Saved t-SNE plot')
    plt.close()

    plt.close()
