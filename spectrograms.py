# -*- coding: utf-8 -*-
"""
@author: F.Rosenthal
"""

from os import listdir
from os import walk
from scipy.io import wavfile
from scipy.signal import spectrogram
from util import plot_spectrogram
import numpy as np
from sklearn.model_selection import train_test_split


def get_filepaths(path="data/"):
    """returns all filepaths to all files in all subdirectories.txt.
    In this project it is needed because of the specific folder structure of the used dataset,
    where one folder contains all files recorded by one speaker.

    Args:
        path (string, optional): Path to root folder of dataset. Defaults to "data/".

    Returns:
        List of strings: Every string containing the filepath to one file in a list.
    """
    folder_names = [x[0] + '/' for x in walk(path) if len(x[0]) > len(path)]
    filepaths = []
    for folder_name in folder_names:
        filepaths.extend([folder_name + filename_folder for filename_folder in listdir(folder_name)])
    return filepaths


def create_all_spectrograms(filepaths, cutoff_value=6000, plot=False, verbose=False):
    """This function calculates all spectrograms to the files specified with their filepaths.
    They are saved in a nested dictionary and assigned to a index.

    It will also save meta data such as the recorded spoken digit, the speaker id,
    the shape of the spectrogram and the filename in a dict.

    All spectrograms will then be zero padded to the longest occuring spectrogram time length.
    The spectrograms will then be flattened across time dimension.
    The resulting dictionary will be returned.

    Args:
        filepaths (list of strings): Filepaths to files.
        cutoff_value (int, optional): Controls the spectrogram size. Frequencies above this value will be cutoff. Defaults to 6000.
        plot (bool, optional): If True, the function will plot a spectrogram every 500 files. Defaults to False.

    Returns:
        dict: A nested dictionary containing meta data of the dataset, as well as all zero padded spectrograms and meta data for each file:
            The returned dictionary is structured as follow:

            spec_dict = {'max_shape': int,    # maximum time length of a sample in dataset used in zero padding
                        'cutoff_value': int,  # used cutoff frequency
                        'cutoff_index': int,  # corresponding cutoff index
                        'specs': {}}          # the nested dict with spectrograms

                        # and in the nested dict, where id is an int and the index corresponding to the specific file:

                        spec_dict['specs'][id] = {'spec': np.array,
                                                'digit': int,
                                                'speaker': int,
                                                'shape': int,  # here only time length is stored
                                                'name': string}
    """
    def print_progress(i, len_filepaths):
        if verbose and i % 500 == 0:
            print(f"{i}/{len_filepaths}", end=" ")

    cutoff_index = int(cutoff_value / 187.5)
    spec_dict = {'max_shape': 0,
                 'cutoff_value': cutoff_value,
                 'cutoff_index': cutoff_index,
                 'specs': {}}
    print("Calculating spectrograms...")
    id = 0
    len_filepaths = len(filepaths)
    for i, wav_fname in enumerate(filepaths):
        print_progress(i=i, len_filepaths=len_filepaths)
        fs, x = wavfile.read(wav_fname)
        f, t, Sxx = spectrogram(x, fs)
        Sxx = Sxx[0:cutoff_index, :]
        f = f[0:cutoff_index]

        if plot and i % 500 == 0:
            plot_spectrogram(t, f, Sxx)

        if Sxx.shape[1] > spec_dict['max_shape']:
            spec_dict['max_shape'] = Sxx.shape[1]

        short_name = wav_fname.split(sep='/')[-1]
        file_digit = int(short_name.split(sep="_")[0])
        speaker_index = int(short_name.split(sep="_")[1])

        spec_dict['specs'][id] = {'spec': Sxx,
                                  'digit': file_digit,
                                  'speaker': speaker_index,
                                  'shape': Sxx.shape[1],
                                  'name': wav_fname}
        id += 1
    print("\nZero padding the spectrograms...")
    max_row_length = spec_dict['max_shape']
    for i, id in enumerate(spec_dict['specs'].keys()):
        print_progress(i=i, len_filepaths=len_filepaths)
        npad = ((0, 0), (0, max_row_length - spec_dict['specs'][id]['spec'].shape[1]))
        spec_dict['specs'][id]['spec'] = np.pad(spec_dict['specs'][id]['spec'],
                                                pad_width=npad,
                                                mode='constant',
                                                constant_values=(0)).flatten('C')
    return spec_dict


def stack_training(spec_dict, train_set, verbose=False):
    """This function prepares the spectrogram dataset for the training SVD.
    It takes the spectrograms in spec_dict indixed by train_set.
    Then all the flattened spectrograms of a digit will be stacked column wise in an numpy nd.array.
    The 10 arrays are stored as values in a dictionary with the digits as keys.

    Args:
        spec_dict (dict): The dictionary returned by create_all_spectrograms
        train_set (list): A list of indices corresponding to the training dataset as returned by sklearn train_test_split().

    Returns:
        dict: The dictionary containing digits (int) as keys and column-wise stacked nd.arrays as values.
                train_stack = {0: nd.array,
                            1: nd.array,
                            2: nd.array,
                            3: nd.array,
                            ...}
    """
    train_stack = {}
    digits = []
    print("Stacking spectrograms in preparation for SVD...")
    for i, id in enumerate(train_set):
        if verbose and i % 500 == 0:
            print(i, end=" ")
        digit = spec_dict['specs'][id]['digit']
        if i == 0 or digit not in digits:
            train_stack[digit] = spec_dict['specs'][id]['spec']
            digits.append(digit)
            del spec_dict['specs'][id]['spec']
        else:
            digit = spec_dict['specs'][id]['digit']
            train_stack[digit] = np.column_stack((train_stack[digit], spec_dict['specs'][id]['spec']))
            del spec_dict['specs'][id]['spec']
    print()
    return train_stack


if __name__ == '__main__':
    path = "data"
    test_size = 0.25
    random_seed = 42
    spec_dict = create_all_spectrograms(get_filepaths(path))
    indices = list(range(100))
    train_set, test_set = train_test_split(indices,
                                           test_size=test_size,
                                           random_state=random_seed)
    train_stack = stack_training(spec_dict, train_set)
