# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 16:17:53 2022

@author: F.Rosenthal, P.Schwarz
"""
import os
import time
from os.path import isfile
import joblib
import pandas as pd

from training import get_svd_path, calc_svd, filter_dataset, estimate_digit
from spectrograms import create_all_spectrograms, stack_training, get_filepaths, train_test_split
from util import format_time, plot_graph, plot_multiline_graph, plot_confusion_matrix, boxplot_residuals


def digit_classifier(k_list=None, size=0, path="data/", verbose=False):
    """
    Trains and tests the classification algorithm to determine digits spoken in the dataset audio files.

    :param k_list: Array of values, that determine how many singular values are used for the classification
    :param size: Amount of test samples to be used in the test run. Default is 50.
    :param path: Path to the file location of the testing data. Default is 'data/'.
    :param verbose: If set to true, the single test results will be printed out during testing.
    """
    if verbose:
        print("Verbose output is activated.")
    else:
        print("Verbose output is deactivated.")

    if k_list is None:
        k_list = [1500]

    split_ratio = 0.25
    random_seed = 42
    filter_key = None
    filter_value = None

    cachepath = os.path.join(os.getcwd(), "cache")
    os.makedirs(cachepath, exist_ok=True)

    spec_filepath = cachepath + "/spectrogram.z"
    svd_filepath = cachepath + "/" + get_svd_path('digit', split_ratio, random_seed)

    if isfile(spec_filepath):
        print('Loading spectrogram data...')
        content = joblib.load(spec_filepath)
    else:
        content = create_all_spectrograms(get_filepaths(path))
        print('Saving spectrogram-file...')
        joblib.dump(content, spec_filepath)

    if filter_key is not None:
        indices = filter_dataset(content['specs'], filter_key)[filter_value]
    else:
        indices = [int(elem) for elem in content['specs'].keys()]
    train_set, test_set = train_test_split(indices, test_size=split_ratio, random_state=random_seed)

    if isfile(svd_filepath):
        print('Loading training data...')
        svd_list = joblib.load(svd_filepath)
    else:
        print('Generating training data...')
        train_stack = stack_training(spec_dict=content, train_set=train_set)
        svd_list = calc_svd(train_stack=train_stack, subset_count=10)
        joblib.dump(svd_list, svd_filepath)

    print('Performing tests...')
    all_error_rates = {}
    for k in k_list:
        eval_dict = {'correct': 0,
                     'incorrect': 0,
                     'cases': {}}

        if size < 1:
            size = len(test_set)
        print(f"Testing {size} samples")
        mean_duration = 0.0
        error_rate_samples = []
        step_times = []
        start = time.time()

        # iterate over test samples
        for i in range(size):
            print(f"### Sample {i + 1} of {size} under Test #", end="\t")
            sample = test_set[i]
            estimated_digit, k, res_list = estimate_digit(svd_list,
                                                          k,
                                                          content['specs'][sample]['spec'])
            test_digit = content['specs'][sample]['digit']

            print(f'Predicted class: {estimated_digit}.',
                  f'Actual class: {test_digit}.', end=" ")

            # write evaluation and update counter for correct or incorrect.
            if estimated_digit == test_digit:
                eval_dict['correct'] += 1
                print("Success!")
            else:
                eval_dict['incorrect'] += 1
                print("We'll get it next time!")

            err_i = eval_dict['incorrect'] / (i + 1)
            error_rate_samples.append(err_i)

            delta = round((time.time() - start))
            start = time.time()
            step_times.append(delta)
            mean_duration = (mean_duration * i + delta) / (i + 1)
            remaining = (size - i) * mean_duration
            print(f"Estimated time remaining: {format_time(remaining)}")

            eval_dict['cases'][i] = {'estimated': estimated_digit,
                                     'actual': test_digit,
                                     'correct': estimated_digit == test_digit,
                                     'error rate': err_i,
                                     'duration': delta,
                                     'residuals': res_list}

        # print and plot the outcome
        plot_graph(step_times, title="Time per Sample", ylabel="Seconds", ground=True)

        all_error_rates[k] = error_rate_samples
        error_rate = eval_dict['incorrect'] / size
        success_rate = eval_dict['correct'] / size
        print(f"Error rate: {error_rate * 100} %\nSuccess rate: {success_rate * 100} %")
        plot_graph(error_rate_samples, title=f"Error Rate Convergence (k={k})", ylim=[0, 1])
        if verbose:
            for key, value in eval_dict.items():
                if key != 'cases':
                    print(key, ' : ', value)

        # Save test results
        metrics = {'samples': size, 'error_rate': error_rate, 'used_EV': k, 'results': eval_dict['cases']}
        df = pd.DataFrame.from_dict(metrics)
        os.makedirs(os.path.join(os.getcwd(), "reports"), exist_ok=True)
        df.to_json(f'reports/number_metrics_{size}_{k}.json')
        df.to_csv('reports/number_metrics.csv', mode="a", index=False, header=True)

        plot_confusion_matrix(metrics)
        boxplot_residuals(metrics)

    plot_multiline_graph(all_error_rates, "Overview Error rates")


if __name__ == "__main__":
    print("__Spoken Digit Classifier by Philipp Schwarz & Fabian Rosenthal__")
    digit_classifier(size=5, path="data/")
