import os
import json
import numpy as np

from models.dvrl import DVRL
from models.evaluator import Evaluator
from utils import get_model_params, load_dataset
from visualize import remove_high_low
from param import args


def data_valuate(file_name, dvrl_instance, cifar10_train, file_dir='./data_values'):
    if os.path.exists(os.path.join(file_dir, file_name)):
        dve_out = json.load(open(os.path.join(file_dir, file_name), 'r'))
        return np.array(dve_out)

    print('save data values')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    data_values = dvrl_instance.get_data_values(cifar10_train)
    dvrl_out = [str(data_values[i]) for i in range(data_values.shape[0])]
    json.dump(dvrl_out, open(os.path.join(file_dir, file_name), 'w'), indent=4)

    return data_values


def visualize_values(dve_out, train_dataset, valid_dataset, test_dataset, noise_rate, figure_name='cifar10'):
    eval_model = Evaluator()
    if not os.path.exists('./visual_files'):
        os.mkdir('./visual_files')
    temp_output = remove_high_low(dve_out, eval_model, train_dataset, valid_dataset, test_dataset, plot=True, data_name=figure_name)
    json.dump(temp_output, open('./visual_files/{}_temp_output.json'.format(figure_name), 'w'), indent=4)


def main(args):
    # 1 数据准备
    train_dataset, valid_dataset, test_dataset = load_dataset(data_name=args.data_name)
    # 2 模型准备
    dict_dvrl = get_model_params(args)
    dvrl = DVRL(train_dataset, valid_dataset, dict_dvrl)

    # 3 训练模型
    print('start training.')
    dvrl.train()

    # 4 数据价值评价
    print('data valuation.')
    file_name = 'dvrl_' + args.data_name + '_train' +str(len()) + '_noise' + str(args.noise_rate) + '.json' if args.noise_rate > 0 else 'dvrl_' + args.data_name + '_train' +str(len(train_dataset)) + '.json'
    dve_out = data_valuate(file_name, dvrl, train_dataset)
    
    # 5 可视化结果保存
    print('visualization.')
    figure_name = '.'.join(file_name.split('.')[:-1])
    visualize_values(dve_out, train_dataset, valid_dataset, test_dataset, args.noise_rate, figure_name)

    print('finished!')


if __name__ == '__main__':
    main(args)
