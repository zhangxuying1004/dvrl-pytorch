from typing import ItemsView
import torchvision
import os
import numpy as np

from data.data_utils import get_random_idx, corrupt_label


def get_subset(dataset, idxs):
    x, y = [], []
    for idx in idxs:
        x_item, y_item = dataset[idx]

        x.append(np.array(x_item))
        y.append(y_item)

    return np.array(x), np.array(y)


def load_cifar(cifar_root, dict_no, noise_rate=0., data_name='cifar10', is_download=False):
    """
    data_name: cifar10
    """
    print('this is {}'.format(data_name))
    if not os.path.exists(cifar_root):
        os.mkdir(cifar_root)
    if data_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root=cifar_root, train=True, download=is_download)
        test_dataset = torchvision.datasets.CIFAR10(root=cifar_root, train=False, download=is_download)
    elif data_name == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root=cifar_root, train=True, download=is_download)
        test_dataset = torchvision.datasets.CIFAR100(root=cifar_root, train=False, download=is_download)
    else:
        print('error data name input: {}!'.format(data_name))
        return

    train_idxs, valid_idxs, test_idxs = get_random_idx(data_name, dict_no, train_dataset, test_dataset)

    x_train, y_train = get_subset(train_dataset, train_idxs)
    x_valid, y_valid = get_subset(train_dataset, valid_idxs)
    x_test, y_test = get_subset(test_dataset, test_idxs)

    if noise_rate > 0:
        y_train, noise_idx = corrupt_label(y_train, noise_rate)
    else:
        noise_idx = np.array([])
    return (x_train, y_train, noise_idx), (x_valid, y_valid), (x_test, y_test)
