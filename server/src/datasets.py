from torchvision import datasets, transforms
import numpy as np


def get_test_dataset():
    data_dir = '../data/mnist/'
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

    return test_dataset


def get_user_group(num_users, dict_users):
    data_dir = '../data/mnist/'
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)

    dict_user = mnist_iid(train_dataset, num_users, dict_users)

    return dict_user


def mnist_iid(dataset, num_users, dict_users):
    num_items = int(len(dataset)/num_users)
    dict_user, all_idxs = [], [i for i in range(len(dataset))]

    # Remove all selected data from randomly choose
    for i in range(num_users):
        if i in dict_users:
            all_idxs = list(set(all_idxs) - dict_users[i])

    dict_user = set(np.random.choice(all_idxs, num_items, replace=False))

    return dict_user
