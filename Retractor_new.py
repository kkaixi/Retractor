# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:48:25 2019

Code for retractor (Nov 2019)

@author: tangk
"""

from PMG.read_data import PMGDataset
from PMG.COM.get_props import get_peaks
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


directory = 'P:\\Data Analysis\\Projects\\Retractor\\'
cutoff = range(100, 1600)
channels = []
table_filters = {'query_list': [['Model', ['Takata','Harmony','Bubble Bum']],
                                ['ATD', ['Y7','Q6']],
                                ['Retractor',[np.nan, 'Marc_4', 'None', 'Caravan']],
                                ['Adjust', [3.75, np.nan]],
                                ['Buckle', ['short_flex', np.nan, 'original']]]}
preprocessing = None

#%% read data

dataset = PMGDataset(directory, channels=channels, cutoff=cutoff, verbose=False)
dataset.table_filters = table_filters
dataset.preprocessing = preprocessing

dataset.get_data([])

#%% plot excursion values
plot_channels = ['Head_Excursion',
                 'Knee_Excursion']
grouped = dataset.table.query('Config==\'sled_213_new_decel\'').groupby(['ATD','Model'])
for ch in plot_channels:
    for grp in grouped:
        subset = grp[1]
        subset['cond'] = subset[['Pulse','Retractor','Buckle']].apply(lambda x: ' '.join(x), axis=1)
        fig, ax = plt.subplots()
        ax = sns.barplot(x='cond', y=ch, data=subset, ax=ax)
        ax.set_title(grp[0])
        plt.show()
        plt.close(fig)