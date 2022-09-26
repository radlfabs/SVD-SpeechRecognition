# -*- coding: utf-8 -*-
"""
@author: P.Schwarz
"""

import numpy as np
import pandas as pd
from numpy import linalg


def get_svd_path(goal, test_ratio, random_seed):
    return "svd_" + goal + "_" + str(test_ratio) + "_" + str(random_seed)


def calc_svd(train_stack, subset_count, verbose=False):
    """
    Calculates the singular value decomposition for the given training data

    :param train_stack: Training data
    :param subset_count: Number of classes the data will be divided in
    :param verbose: If true a console output will be generated for every subset calculation
    :return: List of calculated SVDs
    """
    svd_list = []
    if not verbose:
        print("Calculating SVD on training data...")
    for i in range(subset_count):
        if verbose:
            print("Calculating SVD of training data for subset ", i)
        u, _, _ = linalg.svd(train_stack[i])
        svd_list.append(u)
    return svd_list


def estimate_digit(svd_list, k, test_digit, verbose=False):
    """
    Classifies a sample by a given svd set to determine the most likely digit said in the sample
    :param svd_list: List with SVDs
    :param k: number of singular values used in the calculation
    :param test_digit: test sample
    :param verbose: If true, a log output will be generated for every calculated residuum
    :return: most likly digit, amount of SVs used, list of all calculated residuals
    """
    max_n_sv = svd_list[0].shape[1]
    if k > max_n_sv:
        k = max_n_sv
    if verbose:
        print("\tCalculating residuals:", end=" ")
    residuals = []
    for i in range(10):
        if verbose:
            print(f"#{i}", end=" ")
        test_digit = test_digit.flatten('C')
        Uk = svd_list[i][:, :k]
        residual = linalg.norm((np.identity(len(test_digit)) - Uk @ np.transpose(Uk)) @ test_digit, 2)
        residuals.append(residual)
    return residuals.index(min(residuals)), k, residuals


def filter_dataset(data, filter_key):
    """
    Filters given dataset by a given key and return the indices of filtered entries.
    :param data: Dataset as dict
    :param filter_key: Key from the given dict
    :return: dict with indices assigned to every unique value of the given filter key
    """
    df = pd.DataFrame.from_dict(data, orient='index')
    print(df)
    keys = df[filter_key].unique()
    filtered_data = {}
    for k in keys:
        indices = df.index[df[filter_key] == k].tolist()
        filtered_data[k] = indices
    return filtered_data
