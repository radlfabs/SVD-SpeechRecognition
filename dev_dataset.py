# -*- coding: utf-8 -*-
"""
@author: F.Rosenthal
"""

from shutil import copyfile
from itertools import chain
from pandas import DataFrame
import os
from spectrograms import create_all_spectrograms, get_filepaths


def generate_dev_data():
    """This function takes a random sample of the audioMNIST files and creates a reduced data set 
    for developing and testing purposes.
    It will randomly sample 30 audio files per digit.
    Folder structure will be usable to test the digit classifier.
    """
    path = "data"
    spec_dict = create_all_spectrograms(get_filepaths(path))
    df = DataFrame.from_dict(spec_dict['specs'], orient='index')
    rand_names = []
    for i in range(10):
        rand_names.append(df.loc[df['digit'] == i]['name'].sample(n=30).tolist())
    rand_names_flat = list(chain(*rand_names))
    for i, original in enumerate(rand_names_flat):
        target = 'dev_data\\' + original.split('\\')[1]
        os.makedirs(os.path.join(os.getcwd(), target.split('/')[0]), exist_ok=True)
        copyfile(original, target)