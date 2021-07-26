import torch

from torch.utils.data import DataLoader

from data.dataset import DVDataset
from data.transforms import build_transforms
from param import args


def load_dataset(data_name='cifar10'):
    train_transforms = build_transforms(isTrain=True)
    infer_transforms = build_transforms(isTrain=False)
    
    if data_name == 'cifar10':
        train_dataset = DVDataset(args.cifar10_dir, 'cifar10', dict_no=args.dict_no, data_split='train', trasnform=train_transforms)
        valid_dataset = DVDataset(args.cifar10_dir, 'cifar10', dict_no=args.dict_no, data_split='valid', trasnform=infer_transforms)
        test_dataset = DVDataset(args.cifar10_dir, 'cifar10', dict_no=args.dict_no, data_split='test', trasnform=infer_transforms)
    elif data_name == 'imagenet':
        train_dataset = DVDataset(args.imagenet_dir, 'imagenet', dict_no=args.dict_no, data_split='train', trasnform=train_transforms)
        valid_dataset = DVDataset(args.imagenet_dir, 'imagenet', dict_no=args.dict_no, data_split='valid', trasnform=infer_transforms)
        test_dataset = DVDataset(args.imagenet_dir, 'imagenet', dict_no=args.dict_no, data_split='test', trasnform=infer_transforms)
    else:
        print('data_name error')
        return 
    return train_dataset, valid_dataset, test_dataset


def get_model_params(args):
    """
    提取dvrl模型的参数
    """
    dict_dvrl = {}
    dict_dvrl['dve_lr'] = args.dve_lr
    dict_dvrl['pred_lr'] = args.pred_lr
    dict_dvrl['feat_dim'] = args.feat_dim
    dict_dvrl['category_num'] = args.label_dim
    dict_dvrl['epsilon'] = args.epsilon
    dict_dvrl['threshold'] = args.threshold
    dict_dvrl['outer_iterations'] = args.outer_iterations
    dict_dvrl['inner_iterations'] = args.inner_iterations
    dict_dvrl['batch_size'] = args.batch_size
    dict_dvrl['batch_size_predictor'] = args.mini_batch_size

    return dict_dvrl


def one_hot(label, category_num=10):
    """
    将label tensor转化为one-hot形式
    """
    out = torch.zeros(label.size(0), category_num).long()
    idx = torch.unsqueeze(label, dim=1)
    out.scatter_(dim=1, index=idx, value=1)
    return out
    

def train_eval_model(eval_model, init_checkpoint, subset):
    init_data = torch.load(init_checkpoint)
    eval_model.load_state_dict(init_data)
    eval_model.train()
    
    return eval_model

def infer_eval_model(eval_model, dataset):
    eval_model.eval()
    dataloader = DataLoader(dataset=dataset, batch_size=10)
    
    total_num, correct_num = 0, 0
    for x_batch, y_batch in dataloader:
        total_num += y_batch.shape[0]
        with torch.no_grad():
            y_pred_batch = eval_model(x_batch)   # (batch, category_num)
        correct_num += y_pred_batch.eq(y_batch).cpu().float().sum().item()
    return correct_num / total_num


