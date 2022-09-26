# -*- coding: utf-8 -*-
"""
@author: F.Rosenthal

Summary: This script helps to generate descriptive statistics of the AudioMNIST data set.
"""

import json
import pandas as pd
import pprint

if __name__ == '__main__':
    path = "speakerdata.json"
    with open(path) as json_file:
        data = json.load(json_file)
    df = (pd.DataFrame.from_dict(data=data, orient='index')
          .astype({'age': 'int'})
          .where(lambda x: x.age < 100)
          .assign(age_mean=lambda x: x['age'].mean())
          .assign(age_sd=lambda x: x['age'].std())
          )
    # pprint.pprint(df.head(4))
    pprint.pprint(df.head(1))
    gender_normalized = df['gender'].value_counts() / sum(df['gender'].value_counts())
    pprint.pprint(gender_normalized)
    # male      0.8
    # female    0.2
    # Name: gender
    native_normalized = df['native speaker'].value_counts() / sum(df['native speaker'].value_counts())
    pprint.pprint(native_normalized)
    # no     0.95
    # yes    0.05
    # Name: native speaker
    accent_norm = df['accent'].value_counts() / sum(df['accent'].value_counts())
    pprint.pprint(accent_norm)
    # German                0.666667
    # Chinese               0.050000
    # Italian               0.033333
    # Spanish               0.033333
    # French                0.016667
    # Danish                0.016667
    # Arabic                0.016667
    # South African         0.016667
    # Egyptian_American?    0.016667
    # german                0.016667
    # Brasilian             0.016667
    # English               0.016667
    # Levant                0.016667
    # Madras                0.016667
    # South Korean          0.016667
    # German/Spanish        0.016667
    # Tamil                 0.016667
    # Name: accent, dtype: float64

    # accent_norm[2] = accent_norm[2]
