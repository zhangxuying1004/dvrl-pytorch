import argparse
import os


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='parameters')

        # dataset
        self.parser.add_argument('--imagenet_dir', type=str, default='/zhangxuying/Dataset/imagenet/')
        self.parser.add_argument('--cifar10_dir', type=str, default='/zhangxuying/Dataset/cifar10/')
        self.parser.add_argument('--cifar100_dir', type=str, default='/zhangxuying/Dataset/cifar100/')

        self.parser.add_argument('--image_size', type=tuple, default=(32, 32))
        self.parser.add_argument('--selected_idx_dir', type=str, default='./data_files/data_idx')
        self.parser.add_argument('--dict_no', type=dict, default={'train':4000, 'valid':1000, 'test': 2000})

        self.parser.add_argument('--data_name', type=str, default='cifar10')

        # model
        self.parser.add_argument('--hidden_dim', type=int, default=100, help='dimensions of hidden states')
        self.parser.add_argument('--comb_dim', type=int, default=10, help='dimensions of hidden states after combinding with prediction diff')
        self.parser.add_argument('--layer_number', type=int, default=5, help='number of network layers')
        self.args = self.parser.parse_args()


param = Param()
args = param.args

selected_idx_dir = args.selected_idx_dir
if not os.path.exists(selected_idx_dir):
    os.makedirs(selected_idx_dir)
