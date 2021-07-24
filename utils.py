import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics
from param import args


def get_model_params(args):
    dict_dvrl = {}

    return dict_dvrl


def discover_corrupted_sample(dve_out, noise_idx, noise_rate, plot=True, data_name='cifar10'):
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
        plt.title('Corrupted Sample Discovery of {}'.format(data_name), size=16)
        plt.savefig(os.path.join(args.visual_files, '{}_corrupted_samples.jpg'.format(data_name)))
        plt.show()

    # Returns True Positive Rate of corrupted label discovery
    return output_perf

def remove_high_low(dve_out, eval_model, x_train, y_train,
                    x_valid, y_valid, x_test, y_test,
                    perf_metric='rmspe', plot=True, data_name='cifar10'):
    """Evaluates performance after removing a portion of high/low valued samples.
    Args:
        dve_out: data values
        eval_model: evaluation model (object)
        x_train: training features
        y_train: training labels
        x_valid: validation features
        y_valid: validation labels
        x_test: testing features
        y_test: testing labels
        perf_metric: 'auc', 'accuracy', or 'rmspe'
        plot: print plot or not
    Returns:
        output_perf: Prediction performances after removing a portion of high
                    or low valued samples.
    """

    x_train = np.asarray(x_train)
    y_train = np.reshape(np.asarray(y_train), [len(y_train),])
    x_valid = np.asarray(x_valid)
    y_valid = np.reshape(np.asarray(y_valid), [len(y_valid),])
    x_test = np.asarray(x_test)
    y_test = np.reshape(np.asarray(y_test), [len(y_test),])

    # Sorts samples by data values
    num_bins = 20  # Per 100/20 percentile
    sort_idx = np.argsort(dve_out)
    n_sort_idx = np.argsort(-dve_out)

    # Output Initialization
    if perf_metric in ['auc', 'accuracy']:
        temp_output = np.zeros([2 * num_bins, 2])
    elif perf_metric == 'rmspe':
        temp_output = np.ones([2 * num_bins, 2])

    # For each percentile bin
    for itt in range(num_bins):

        # 1. Remove least valuable samples first
        new_x_train = x_train[sort_idx[int(itt*len(x_train[:, 0])/num_bins):], :]
        new_y_train = y_train[sort_idx[int(itt*len(x_train[:, 0])/num_bins):]]

        if len(np.unique(new_y_train)) > 1:
            eval_model.fit(new_x_train, new_y_train)

            y_valid_hat = eval_model.predict_proba(x_valid)
            y_test_hat = eval_model.predict_proba(x_test)

            temp_output[itt, 0] = metrics.accuracy_score(y_valid,
                                                        np.argmax(y_valid_hat,
                                                                axis=1))
            temp_output[itt, 1] = metrics.accuracy_score(y_test,
                                                        np.argmax(y_test_hat,
                                                                axis=1))

        # 2. Remove most valuable samples first
        new_x_train = x_train[n_sort_idx[int(itt*len(x_train[:, 0])/num_bins):], :]
        new_y_train = y_train[n_sort_idx[int(itt*len(x_train[:, 0])/num_bins):]]

        if len(np.unique(new_y_train)) > 1:
            eval_model.fit(new_x_train, new_y_train)

            y_valid_hat = eval_model.predict_proba(x_valid)
            y_test_hat = eval_model.predict_proba(x_test)

            temp_output[num_bins + itt, 0] = \
                metrics.accuracy_score(y_valid, np.argmax(y_valid_hat, axis=1))
            temp_output[num_bins + itt, 1] = \
                metrics.accuracy_score(y_test, np.argmax(y_test_hat, axis=1))

    # Plot graphs
    if plot:

        # Defines x-axis
        num_x = int(num_bins/2 + 1)
        x = [a*(1.0/num_bins) for a in range(num_x)]

        # Prediction performances after removing high or low values
        plt.figure(figsize=(6, 7.5))
        plt.plot(x, temp_output[:num_x, 1], 'o-')
        plt.plot(x, temp_output[num_bins:(num_bins+num_x), 1], 'x-')

        plt.xlabel('Fraction of Removed Samples', size=16)
        plt.ylabel('Accuracy', size=16)
        plt.legend(['Removing low value data', 'Removing high value data'],
                prop={'size': 16})
        plt.title('Remove High/Low Valued Samples of {}'.format(data_name), size=16)
        plt.savefig(os.path.join(args.visual_files, '{}_remove_partial_samples.jpg'.format(data_name)))
        plt.show()

    return temp_output
