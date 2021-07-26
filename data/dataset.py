import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict

from data.datasets.cifar import load_cifar
from data.datasets.imagenet import load_subimagenet
from models.encoder import Encoder

from param import args


dataname2func: Dict[str, object] = {
    'cifar10': load_cifar,
    'cifar100': load_cifar,
    'imagenet': load_subimagenet
}

class DVDataset(Dataset):
    def __init__(self, dataset_root, data_name, dict_no={'train':4000, 'valid':2000, 'test':1000}, data_split='train', noise_rate=0., trasnform=None) -> None:
        super(Dataset, self).__init__()
        self.dataset_root = dataset_root
        self.data_name = data_name
        self.dict_no = dict_no
        self.data_split = data_split
        self.transform = trasnform
        self.noise_rate = noise_rate

        # self.encoder = Encoder()

        (x_train, y_train, noise_idx), (x_valid, y_valid), (x_test, y_test) = \
            dataname2func[self.data_name](dataset_root, self.dict_no, self.noise_rate, self.data_name)
        if data_split == 'train':
            self.images, self.targets, self.noise_idx = x_train, y_train, noise_idx
        elif data_split == 'valid':
            self.images, self.targets = x_valid, y_valid
        else:
            self.images, self.targets = x_test, y_test
    
        print('images:', self.images.shape)
        print('targets:', self.targets.shape)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> tuple:
        image = self.images[index]
        label = self.targets[index]
        
        if self.transform is not None:
            image = self.transform(image)
        # label = torch.from_numpy(np.array(label))
        label = torch.tensor(label)
        return (image, label)


class DVSubset(Dataset):
    def __init__(self, dataset, idxs):
        super(Dataset, self).__init__()
        self.dataset = dataset
        self.idxs = idxs
        
        self.x_list, self.y_list = [], []
        for idx in idxs:
            x_item, y_item = dataset[idx]
            self.x_list.append(x_item)
            self.y_list.append(y_item)
    
        print('this subset:', len(self.x_list))
    
    def __len__(self):
        return len(self.x_list)
    
    def __getitem__(self, index: int):
        image = self.x_list[index]
        label = self.y_list[index]
        return (image, label)