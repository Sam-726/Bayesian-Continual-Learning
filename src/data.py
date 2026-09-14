from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

@dataclass
class Task:
    train: Subset
    test: Subset
    classes: Tuple[int, int]

def _subset_for_classes(dataset, classes):
    targets = torch.as_tensor(dataset.targets)
    idx = torch.where((targets == classes[0]) | (targets == classes[1]))[0].tolist()
    return Subset(dataset, idx)

def get_split_mnist(root: str = "./data", num_workers: int = 0) -> List[Task]:
    tfm = transforms.ToTensor()
    train = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root=root, train=False, download=True, transform=tfm)
    pairs = [(0,1),(2,3),(4,5),(6,7),(8,9)]
    return [Task(_subset_for_classes(train,p), _subset_for_classes(test,p), p) for p in pairs]

def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
