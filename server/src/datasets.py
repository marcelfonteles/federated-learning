from torchvision import datasets, transforms
import numpy as np


def get_dataset():
    data_dir = '../data/mnist/'
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

    return test_dataset
