from typing import Dict
import cv2
import os
import json
import numpy as np

from param import args


def read_image(img_path, size=(32, 32)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    return img


def get_random_idx(data_name, dict_no, train_dataset=None, test_dataset=None):
    train_idxs, valid_idxs, test_idxs = [], [], []
    idx_file_name = data_name + '_idx.json'
    idx_file_path = os.path.join(args.selected_idx_dir, idx_file_name)
    if os.path.exists(idx_file_path):
        print('load random idx from json file')
        cifar_idxs = json.load(open(idx_file_path, 'r'))
        train_idxs = cifar_idxs['train']
        valid_idxs = cifar_idxs['valid']
        test_idxs = cifar_idxs['test']
    else:
        print('random select')
        if (train_dataset is not None) and (test_dataset is not None):
            train_idxs = np.random.permutation(len(train_dataset))
            valid_idxs = train_idxs[:dict_no['valid']]
            train_idxs = train_idxs[dict_no['valid']:(dict_no['train']+dict_no['valid'])]
            test_idxs = np.random.permutation(len(test_dataset))[:dict_no['test']]
        elif (train_dataset is not None) and (test_dataset is None):
            total_idxs = np.random.permutation(len(train_dataset))
            valid_idxs = total_idxs[:dict_no['valid']]
            train_idxs = total_idxs[dict_no['valid']:(dict_no['train']+dict_no['valid'])]
            test_idxs = total_idxs[(dict_no['train']+dict_no['valid']):(dict_no['train']+dict_no['valid']+dict_no['test'])]
        else:
            print('error! please input dataset params in get_random_idx function.')
            return

        cifar_idxs = {
            'train': train_idxs.tolist(),
            'valid': valid_idxs.tolist(),
            'test': test_idxs.tolist()
        }
        json.dump(cifar_idxs, open(idx_file_path, 'w'), indent=4)
        print('save random idx to json file')
    
    return train_idxs, valid_idxs, test_idxs
