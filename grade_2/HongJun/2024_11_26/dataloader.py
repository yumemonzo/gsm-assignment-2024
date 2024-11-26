import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import random_split

def get_transform():
    transform = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomCrop(size=(32, 32), padding=4),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform

def get_datasets(transform):
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def split_dataset(dataset, split_size):
    size_A = int(split_size * len(dataset))
    size_B = len(dataset) - size_A
    dataset_A, dataset_B = random_split(dataset, [size_A, size_B])
    return dataset_A, dataset_B
