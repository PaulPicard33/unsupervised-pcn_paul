from torch.utils import data
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, random_split, DataLoader

from pcn import utils
import numpy as np
import torch

class MNIST(datasets.MNIST):
    def __init__(self, train, size=None, n_classes=None, normalize=False):
        transform = _get_transform(normalize=normalize, mean=(0.1307), std=(0.3081))
        super().__init__("./data/mnist", download=True, transform=transform, train=train)
        if n_classes is not None:
            self._select(n_classes)
        if size is not None:
            self._reduce(size)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        data = _to_vector(data)
        return data, target

    def _select(self, n_classes):
        N = len(self.data)
        indices = torch.zeros(N, dtype = bool)
        for c in range(n_classes):
            idx = (self.targets == c)
            indices = indices | idx
        self.data = self.data[indices]
        self.targets = self.targets[indices]

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]

class CIFAR10(datasets.CIFAR10):
    def __init__(self, train, size=None, n_classes=None, normalize=False):
        transform = _get_transform(normalize=normalize, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        super().__init__("./data/cifar10", download=True, transform=transform, train=train)
        if n_classes is not None:
            self._select(n_classes)
        if size is not None:
            self._reduce(size)

    def __getitem__(self, index): # get a single sample
        data, target = super().__getitem__(index)
        data = _to_vector(data) 
        return data, target

    def _select(self, n_classes):
        N = len(self.data)
        indices = torch.zeros(N, dtype = bool)
        for c in range(n_classes):
            idx = (self.targets == c)
            indices = indices | idx
        self.data = self.data[indices]
        self.targets = self.targets[indices]

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]

class FashionMNIST(datasets.FashionMNIST):
    def __init__(self, train, size=None, n_classes=None, normalize=False):
        transform = _get_transform(normalize=normalize, mean=(0.5), std=(0.5))
        super().__init__("./data/fmnist", download=True, transform=transform, train=train)
        if n_classes is not None:
            self._select(n_classes)
        if size is not None:
            self._reduce(size)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        data = _to_vector(data)
        return data, target

    def _select(self, n_classes):
        N = len(self.data)
        indices = torch.zeros(N, dtype = bool)
        for c in range(n_classes):
            idx = (self.targets == c)
            indices = indices | idx
        self.data = self.data[indices]
        self.targets = self.targets[indices]

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]


def get_dataloader(dataset, batch_size, worker_init_fn, generator, device=None): 
    # make batches of samples individually obtained with __getitem__
    dataloader = data.DataLoader(
        dataset, 
        batch_size, 
        shuffle=True, 
        drop_last=True, 
        worker_init_fn=worker_init_fn, 
        generator=generator) 
    return list(map(lambda b: _preprocess_batch(b, device), dataloader)) # list of (data, label) batches

def _preprocess_batch(batch, device=None):
    if device is None:
        batch[0] = utils.set_tensor(batch[0])
    else:
        batch[0] = utils.set_tensor(batch[0], device)
    return (batch[0], batch[1])


def _get_transform(normalize=True, mean=(0.5), std=(0.5)):
    transform = [transforms.ToTensor()]
    if normalize:
        transform += [transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(transform)

def _to_vector(data):
    return data.flatten()

def get_fmnist_dataloaders(batch_size=1024, subset_size=None):
    """
    Charge le dataset Fashion-MNIST. 
    subset_size permet de réduire drastiquement la taille pour les tests locaux.
    """
    # Ajout de 2 pixels de bordure pour obtenir du 32x32
    transform_fmnist_padded = transforms.Compose([
        transforms.Pad(2), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])

    fmnist_full = torchvision.datasets.FashionMNIST(
        root='./data/fmnist', train=True, download=True, transform=transform_fmnist_padded
    ) #

    # Réduction du dataset pour les tests locaux
    if subset_size is not None:
        fmnist_full = Subset(fmnist_full, range(subset_size))
        train_size = int(0.8 * len(fmnist_full))
        val_size = len(fmnist_full) - train_size
    else:
        # Séparation standard (50k train / 10k val)[cite: 1]
        train_size = 50000
        val_size = 10000

    fmnist_train, fmnist_val = random_split(fmnist_full, [train_size, val_size]) #[cite: 1]

    train_loader = DataLoader(fmnist_train, batch_size=batch_size, shuffle=True) #[cite: 1]
    val_loader = DataLoader(fmnist_val, batch_size=batch_size, shuffle=False) #[cite: 1]

    return {"train": train_loader, "val": val_loader, "input_dim": 1024, "shape": (1, 32, 32)}
def get_CIFAR10_dataloaders(batch_size=1024, subset_size=None):
    """
    Charge le dataset Fashion-MNIST. 
    subset_size permet de réduire drastiquement la taille pour les tests locaux.
    """
    
    fmnist_full = torchvision.datasets.CIFAR10(
        root='./data/cifar10', train=True, download=True 
    ) #

    # Réduction du dataset pour les tests locaux
    if subset_size is not None:
        fmnist_full = Subset(fmnist_full, range(subset_size))
        train_size = int(0.8 * len(fmnist_full))
        val_size = len(fmnist_full) - train_size
    else:
        # Séparation standard (50k train / 10k val)[cite: 1]
        train_size = 50000
        val_size = 10000

    fmnist_train, fmnist_val = random_split(fmnist_full) #[cite: 1]

    train_loader = DataLoader(fmnist_train, batch_size=batch_size, shuffle=True) #[cite: 1]
    val_loader = DataLoader(fmnist_val, batch_size=batch_size, shuffle=False) #[cite: 1]

    return {"train": train_loader, "val": val_loader, "input_dim": 1024, "shape": (1, 32, 32)}