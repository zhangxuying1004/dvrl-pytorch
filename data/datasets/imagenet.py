import os
import json
from tqdm import tqdm
import numpy as np
import random

from data.data_utils import read_image, get_random_idx, corrupt_label


def check_subcats(folder_dirs):
    sign = True
    for folder_dir in folder_dirs:
        if len(os.listdir(folder_dir)) < 1300:
            sign = False
            break
    return sign


def subcategories_prepare(imagenet_dir, subcats_file='./data_files/subcats/imagenet_subcats.json'):
    if os.path.exists(subcats_file):
        subcats = json.load(open(subcats_file, 'r'))
        return subcats

    while True:
        categories = os.listdir(os.path.join(imagenet_dir, 'train2012'))
        subcats = random.sample(categories, 10)
        folder_dirs = [os.path.join(imagenet_dir, 'train2012', folder_name) for folder_name in subcats]
        if check_subcats(folder_dirs):
            json.dump(subcats, open(subcats_file, 'w'), indent=4)
            return subcats
    

def label_prepare(imagenet_dir, category_folders, isSubData=False):
    folder2label = {}

    if isSubData:
        folder2label = {category_folders[i]:i  for i in range(len(category_folders))}
    else:
        dict_train_labels_path = os.path.join(imagenet_dir, 'imagenet2012_train_labels.json')
        dict_train_labels = json.load(open(dict_train_labels_path, 'r'))

        folder2label = {folder_name: int(dict_train_labels[folder_name]['label_idx']) for folder_name in category_folders}
    
    return folder2label


def assert_data_dir(data_file):
    data_name = data_file.split('/')[-1]
    data_dir = data_file.split(data_name)[0]
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


def load_subimagenet(imagenet_dir, dict_no, noise_rate=0., data_name='imagenet', data_file='./data_files/data_cache/subimagenet.npz'):
    if os.path.exists(data_file):
        print('loading subimagenet from {}'.format(data_file))
        data = np.load(data_file)
        images = data['images']
        targets = data['targets']
    else:
        images, targets = [], []
        subcats = subcategories_prepare(imagenet_dir)
        # label
        folder2label = label_prepare(imagenet_dir, subcats, isSubData=True)

        for folder_name in subcats:

            img_dir = os.path.join(imagenet_dir, 'train2012', folder_name)
            img_names = os.listdir(img_dir)
            for i in tqdm(range(len(img_names))):
                img_name = img_names[i]
                img_path = os.path.join(img_dir, img_name)
                img = read_image(img_path)
                images.append(img)
            
            targets += [folder2label[folder_name]] * len(img_names)
            print(len(targets))

        images = np.array(images)
        targets = np.array(targets)
        
        assert_data_dir(data_file)
        np.savez(data_file, images=images, targets=targets)
        print('subimagenet is saved to {}'.format(data_file))

    print(len(images))
    print(len(targets))

    train_idxs, valid_idxs, test_idxs = get_random_idx(data_name, dict_no, images)

    x_valid = images[valid_idxs]
    x_train = images[train_idxs]
    x_test = images[test_idxs]

    y_valid = targets[valid_idxs].flatten()
    y_train = targets[train_idxs].flatten()
    y_test = targets[test_idxs].flatten()

    if noise_rate > 0:
        y_train, noise_idx = corrupt_label(y_train, noise_rate)
    else:
        noise_idx = np.array([])
    return (x_train, y_train, noise_idx), (x_valid, y_valid), (x_test, y_test)


# def load_subvalid():
#     x_valid, y_valid = [], []
#     valid_img_dir = 'subimagenet/val2012'
#     img_names = os.listdir(valid_img_dir)

#     for i in tqdm(range(len(img_names))):
#         img_name = img_names[i]
#         label_idx = int(img_name.split('_')[0])
#         y_valid.append(label_idx)

#         img_path = os.path.join(valid_img_dir, img_name)
#         img = read_image(img_path)
#         x_valid.append(img)

#     print(len(x_valid))
#     print(len(y_valid))
#     return x_valid, y_valid
