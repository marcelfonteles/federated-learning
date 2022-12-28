from torchvision import datasets, transforms
import numpy as np


def get_dataset(dataset, num_users):
    if dataset == 'mnist':
        data_dir = '../data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

        # user_group = mnist_iid(train_dataset, num_users)

    elif dataset == 'cifar10':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

    return train_dataset, test_dataset #, user_group


# The data selection from dataset is made by the server. This is one way to guarantee that all clients have different
# data. In real world all clients will have they own piece of data and the server will not see any of that data.
def mnist_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_user, all_idxs = [], [i for i in range(len(dataset))]
    dict_user = set(np.random.choice(all_idxs, num_items, replace=False))
    return dict_user
