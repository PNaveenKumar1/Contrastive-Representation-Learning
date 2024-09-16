import torch
import numpy as np


# 'cuda' device for supported nvidia GPU, 'mps' for Apple silicon (M1-M3)
device = torch.device(
    'cuda:6' if torch.cuda.is_available() else 'mps'\
        if torch.backends.mps.is_available() else 'cpu')


def from_numpy(
        x: np.ndarray,
        dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    X =  torch.tensor(x, dtype=dtype)
    # x = torch.from_numpy(x).to(dtype)
    return X.to(device)
    # raise NotImplementedError('Convert numpy array to torch tensor here and send to device')


def to_numpy(x: torch.Tensor) -> np.ndarray:
    

    if x.is_cuda:
        X = x.cpu().detach().numpy()

    else:
        X = x.detach().numpy()

    # raise NotImplementedError('Convert torch tensor to numpy array here')
    return X
    # HINT: if using GPU, move the tensor to CPU before converting to numpy
