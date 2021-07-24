from data.dataset import DVDataset
from data.transforms import build_transforms
from models.dvrl import DVRL
from utils import get_model_params
from param import args

from torch.utils.data import DataLoader


def test():
    train_transforms = build_transforms(isTrain=True)

    cifar10_train = DVDataset(args.cifar10_dir, 'cifar10', dict_no=args.dict_no, data_split='train', trasnform=train_transforms)
    # imagenet_train = DVDataset(args.imagenet_dir, 'imagenet', dict_no=args.dict_no, data_split='train', trasnform=train_transforms)
    dataset = cifar10_train
    # dataset = imagenet_train

    dataloader = DataLoader(
        dataset,
        batch_size=10
    )
    for x, y in dataloader:
        
        print(x.shape)
        print(y)
        print(y.shape)
        idx = [1, 2, 5]
        mini_x = x[idx]
        print(mini_x.shape)
        break


def main():
    # load dataset
    train_transforms = build_transforms(isTrain=True)
    infer_transforms = build_transforms(isTrain=False)
    cifar10_train = DVDataset(args.cifar10_dir, 'cifar10', dict_no=args.dict_no, data_split='train', trasnform=train_transforms)
    cifar10_valid = DVDataset(args.cifar10_dir, 'cifar10', dict_no=args.dict_no, data_split='valid', trasnform=infer_transforms)
    # cifar10_test = DVDataset(args.cifar10_dir, 'cifar10', dict_no=args.dict_no, data_split='test', trasnform=infer_transforms)
    # imagenet_train = DVDataset(args.imagenet_dir, 'imagenet', dict_no=args.dict_no, data_split='train', trasnform=train_transforms)

    # build model
    dict_dvrl = get_model_params(args)
    dvrl = DVRL(cifar10_train, cifar10_valid, dict_dvrl)

    # train model
    dvrl.train()

    # data valuation
    dvrl.get_data_values(cifar10_train)


if __name__ == '__main__':
    test()