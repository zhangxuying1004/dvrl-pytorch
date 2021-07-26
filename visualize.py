import os
import torch
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from copy import copy

from data.dataset import DVSubset
from utils import train_eval_model, infer_eval_model
from param import args


def remove_high_low(dve_out, eval_model, train_dataset, valid_dataset, test_dataset, plot=True, figure_name='cifar10'):
    """Evaluates performance after removing a portion of high/low valued samples.
    Args:
        dve_out: data values
        eval_model: evaluation model (object)
        train_dataset: training features
        valid_dataset: validation features
        test_dataset: testing features
        plot: print plot or not, figure_name
    Returns:
        output_perf: Prediction performances after removing a portion of high
                    or low valued samples.
    """

    # Sorts samples by data values
    num_bins = 20  # Per 100/20 percentile
    sort_idx = np.argsort(dve_out)
    n_sort_idx = np.argsort(-dve_out)

    # 保存eval模型的初始状态
    init_checkpoint = './eval.pth'
    torch.save(eval_model.state_dict(), init_checkpoint)

    # Output Initialization
    temp_output = {
        'rm_low': {'valid': [], 'test': []},
        'rm_high': {'valid': [], 'test': []},
    }

    # For each percentile bin
    for itt in tqdm(range(num_bins)):

        # 1. Remove least valuable samples first
        rm_low_subset = DVSubset(train_dataset, sort_idx[int(itt*len(train_dataset)/num_bins):])
        rm_low_eval_model = train_eval_model(copy(eval_model), init_checkpoint, rm_low_subset)

        rm_low_valid_score = infer_eval_model(rm_low_eval_model, valid_dataset)
        rm_low_test_score = infer_eval_model(rm_low_eval_model, test_dataset)

        temp_output['rm_low']['valid'].append(rm_low_valid_score)
        temp_output['rm_low']['test'].append(rm_low_test_score)

        # 2. Remove most valuable samples first
        rm_high_subset = DVSubset(train_dataset, n_sort_idx[int(itt*len(train_dataset)/num_bins):])
        rm_high_eval_model = train_eval_model(copy(eval_model), init_checkpoint, rm_high_subset)

        rm_high_valid_score = infer_eval_model(rm_high_eval_model, valid_dataset)
        rm_high_test_score = infer_eval_model(rm_high_eval_model, test_dataset)

        temp_output['rm_high']['valid'].append(rm_high_valid_score)
        temp_output['rm_high']['test'].append(rm_high_test_score)

    # Plot graphs
    if plot:
        # Defines x-axis
        num_x = int(num_bins/2 + 1)
        x = [a*(1.0/num_bins) for a in range(num_x)]

        # Prediction performances after removing high or low values
        plt.figure(figsize=(6, 7.5))
        plt.plot(x, temp_output['rm_low']['test'], 'o-')
        plt.plot(x, temp_output['rm_high']['test'], 'x-')

        plt.xlabel('Fraction of Removed Samples', size=16)
        plt.ylabel('Accuracy', size=16)
        plt.legend(['Removing low value data', 'Removing high value data'],
                prop={'size': 16})
        plt.title('Remove High/Low Valued Samples of {}'.format(figure_name), size=16)
        plt.savefig(os.path.join(args.visual_files, '{}_remove_partial_samples.jpg'.format(figure_name)))
        plt.show()

    return temp_output


def discover_corrupted_sample(dve_out, noise_idx, noise_rate, plot=True, figure_name='cifar10'):
    """Reports True Positive Rate (TPR) of corrupted label discovery.
    Args:
        dve_out: data values
        noise_idx: noise index
        noise_rate: the ratio of noisy samples
        plot: print plot or not
    Returns:
        output_perf: True positive rate (TPR) of corrupted label discovery (per 5 percentiles)
    """
    # Sorts samples by data values
    num_bins = 20  # Per 100/20 percentile
    sort_idx = np.argsort(dve_out)

    # Output initialization
    output_perf = np.zeros([num_bins,])

    # For each percentile
    for itt in range(num_bins):
        # from low to high data values
        output_perf[itt] = len(np.intersect1d(sort_idx[:int((itt+1)* \
                                len(dve_out)/num_bins)], noise_idx)) \
                                / len(noise_idx)

    # Plot corrupted label discovery graphs
    if plot:
        # Defines x-axis
        num_x = int(num_bins/2 + 1)
        x = [a*(1.0/num_bins) for a in range(num_x)]

        # Corrupted label discovery results (dvrl, optimal, random)
        y_dvrl = np.concatenate((np.zeros(1), output_perf[:(num_x-1)]))
        y_opt = [min([a*((1.0/num_bins)/noise_rate), 1]) for a in range(num_x)]
        y_random = x

        plt.figure(figsize=(6, 7.5))
        plt.plot(x, y_dvrl, 'o-')
        plt.plot(x, y_opt, '--')
        plt.plot(x, y_random, ':')
        plt.xlabel('Fraction of data Inspected', size=16)
        plt.ylabel('Fraction of discovered corrupted samples', size=16)
        plt.legend(['DVRL', 'Optimal', 'Random'], prop={'size': 16})
        plt.title('Corrupted Sample Discovery of {}'.format(figure_name), size=16)
        plt.savefig(os.path.join(args.visual_files, '{}_corrupted_samples.jpg'.format(figure_name)))
        plt.show()

    # Returns True Positive Rate of corrupted label discovery
    return output_perf