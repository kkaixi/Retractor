# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:30:50 2018

@author: tangk
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from copy import deepcopy
from scipy.signal import medfilt
from PMG.read_data import initialize
from PMG.COM.arrange import *
from PMG.COM.plotfuns import *
from PMG.COM.get_props import *

directory = 'P:\\Data Analysis\\Projects\\Retractor\\'
cutoff = range(100,2300)
channels = ['12HEAD0000Y7ACXA',
            '14HEAD0000Y7ACXA',
            '16HEAD0000Y7ACXA',
            '12CHST0000Y7ACXC',
            '14CHST0000Y7ACXC',
            '16CHST0000Y7ACXC',
            '12CHST0000Y7DSXB',
            '14CHST0000Y7DSXB',
            '16CHST0000Y7DSXB',
            '12PELV0000Y7ACXA',
            '14PELV0000Y7ACXA',
            '16PELV0000Y7ACXA']
drop = ['SE17-0039_3',
        'SE17-0018_2'] # mifold in vehicle

table, t, chdata = initialize(directory,channels, cutoff, drop=drop)
t_, chdata_ = deepcopy(t), deepcopy(chdata)
#%% preprocessing
table['Pos'] = table['Pos'].astype(int).astype(str)

# align all data to the peak onset, defined as 10ms before the first time at which the acceleration exceeds 10% of its peak
# booleans are median filtered with a kernel size of 11 to account for noise
medfilt11 = partial(medfilt, kernel_size=11)

onsets = {'12HEAD0000Y7ACXA': 'min',
          '14HEAD0000Y7ACXA': 'min',
          '16HEAD0000Y7ACXA': 'min'}
onset = pd.DataFrame(index=chdata.index)
for j in chdata.columns:
    if j in onsets:
        if onsets[j]=='min':
            onset['Min_' + j] = chdata[j].apply(lambda x: get_onset_to_min(x, filt=medfilt11))
            continue
        elif onsets[j]=='max':
            onset['Max_' + j] = chdata[j].apply(lambda x: get_onset_to_max(x, filt=medfilt11))
            continue
    
    positive_peak = chdata[j].apply(max) > chdata[j].apply(min).abs()
    n = len(chdata)
    n_positive = positive_peak.sum()
    n_negative = n - n_positive
    
    if n_positive > n/2:
        onset['Max_' + j] = chdata[j].apply(lambda x: get_onset_to_max(x, filt=medfilt11))
    elif n_negative > n/2:
        onset['Min_' + j] = chdata[j].apply(lambda x: get_onset_to_min(x, filt=medfilt11))
    elif n_positive==n_negative:
        print('Channel {} has an even number of positive and negative peaks!'.format(j))
        break
        
for j in onset.columns:
    for i in onset.index:
        onset_ij = onset.at[i, j]
        if not np.isnan(onset_ij):
            tstart = int(onset_ij)-100
            if tstart < 0:
                tstart = 0
            chdata.loc[[i], [j[4:]]] = chdata.loc[[i], [j[4:]]].applymap(lambda x: x[tstart:])

    
min_len = chdata.applymap(len).min().min()
chdata = chdata.applymap(lambda x: x[:min_len])
t = t[:min_len]

#%% Compare of in-vehicle vs. bench + original buckle + vehicle retractor vs. bench + flexible buckle + vehicle retractor 
plot_channels = [['12CHST0000Y7ACXC','14CHST0000Y7ACXC','16CHST0000Y7ACXC']]
#subset = (table.query('Model==\'Turbo Booster\'')
#               .table.query_list('Buckle',['Original','Short Flex','Vehicle'])
#               .table.query_list('Retractor', ['Marc_4','None','Vehicle']))


subset = (table.query('Model==\'Harmony\'')
               .table.query_list('Buckle',['Short Flex', 'Original'])
               .table.query_list('Retractor',['Vehicle','Marc_4'])) 


subset['Cond'] = subset[['Retractor','Buckle']].apply(lambda x: '+'.join(x), axis=1)
grouped = subset.groupby('Cond')

for ch in plot_channels:
    fig, ax = plt.subplots()
    for grp in grouped:
        x = {grp[0]: merge_columns(chdata.loc[grp[1].index, ch])}
        ax = plot_overlay(ax, t, x)
    ax = set_labels(ax, {'legend': {}})
    ax.set_ylim([-70, 10])
        
#%% plot bands
plot_channels = [['12HEAD0000Y7ACXA','14HEAD0000Y7ACXA','16HEAD0000Y7ACXA'],
                 ['12CHST0000Y7ACXC','14CHST0000Y7ACXC','16CHST0000Y7ACXC'],
                 ['12PELV0000Y7ACXA','14PELV0000Y7ACXA','16PELV0000Y7ACXA']]  


#subset = (table.query('(Retractor==\'Marc_4\' and Buckle==\'Original\') or (Retractor==\'Vehicle\' and Buckle==\'Vehicle\')')
#               .table.query_list('Speed', [48, np.nan]))


subset = pd.concat([table.query('Retractor==\'Marc_4\' and Buckle==\'Original\''),
                    table.query('Retractor==\'Vehicle\' and Buckle==\'Vehicle\' and Model==\'Harmony\'')], axis=0)

#subset = pd.concat([table.query('Retractor==\'Marc_4\' and Buckle==\'Original\''),
#                    table.query('Retractor==\'Vehicle\' and Buckle==\'Vehicle\' and Speed==48')], axis=0)


#subset = pd.concat([table.query('Retractor==\'Vehicle\'').table.query_list('Buckle',['Original','Short Flex']),
#                    table.query('Retractor==\'Vehicle\' and Buckle==\'Vehicle\' and Speed==48')], axis=0)


subset['Cond'] = subset[['Retractor','Buckle']].apply(lambda x: '+'.join(x), axis=1)



grouped = subset.groupby('Cond')

for ch in plot_channels:
    fig, ax = plt.subplots()
    legend_labels = []
    for grp in grouped:
        legend_labels.append(grp[0])
        # merge data into one Series
        x = pd.DataFrame({'ch': np.concatenate([chdata.at[i, [j for j in ch if subset.at[i, 'Pos'] in j][0]] for i in grp[1].index]),
                          't': np.tile(t, len(grp[1])),
                          'Cond': np.repeat(grp[1]['Cond'].values, len(t))})   
#       if n<=3, plot individual tests!         
        if len(grp[1])<=3:
            ax = sns.lineplot(x='t', y='ch', estimator=None, data=x, ci=None, ax=ax)
        else:
            ax = sns.lineplot(x='t', y='ch', data=x, ax=ax, linewidth=0.5, linestyle='--')
    ax.legend(ax.lines, legend_labels)
    ax.set_title(ch)
