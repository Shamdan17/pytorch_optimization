# cifar10 data with transforms
import torch
import torchvision
from torchvision import datasets, transforms


def get_data():
    # Create data
    train_dataset = datasets.CIFAR10(
        root="../../data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.AutoAugment(
                    torchvision.transforms.AutoAugmentPolicy.CIFAR10
                ),
                torchvision.transforms.ToTensor(),
            ]
        ),
    )

    val_dataset = datasets.CIFAR10(
        root="../../data",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        ),
    )

    return train_dataset, val_dataset
