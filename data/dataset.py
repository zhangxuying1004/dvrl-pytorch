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
    def __init__(self, dataset_root, data_name, dict_no={'train':4000, 'valid':2000, 'test':1000}, data_split='train', trasnform=None) -> None:
        super(Dataset, self).__init__()
        self.dataset_root = dataset_root
        self.data_name = data_name
        self.dict_no = dict_no
        self.data_split = data_split
        self.transform = trasnform

        # self.encoder = Encoder()

        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = \
            dataname2func[self.data_name](dataset_root, self.dict_no, self.data_name)
        if data_split == 'train':
            self.images, self.targets = x_train, y_train
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
