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
from PMG.COM.easyname import *

directory = 'P:\\Data Analysis\\Projects\\Retractor\\'
cutoff = range(100,2300)
channels = ['S0SLED000000ACXD',
            '10CVEHCG0000ACXD',
            '12HEAD0000Y7ACXA',
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
drop = ['SE17-0039_3', 'SE17-0018_2']

table, t, chdata = initialize(directory,channels, cutoff, drop=drop)
chdata.at['SE18-0004_2','S0SLED000000ACXD'] = -chdata.at['SE18-0004_2','S0SLED000000ACXD']
chdata['10CVEHCG0000ACXD'] = chdata['10CVEHCG0000ACXD'].apply(lambda x: -x)
#%%
table['Pos'] = table['Pos'].astype(int).astype(str)
t_, chdata_ = deepcopy(t), deepcopy(chdata)

#%% assign a colour to all combinations of retractor, buckle, and speed
groups = (table.table.query_list('Retractor',['Marc_4','Vehicle','None'])
               .table.query_list('Buckle',['Original','Short Flex','Vehicle'])[['Retractor','Buckle','Speed']]
               .replace(np.nan, '')
               .drop_duplicates()
               .apply(tuple, axis=1)
               .values)
colors = sns.color_palette()
plot_specs = {i: c for i, c in zip(groups, colors)}
table['Color'] = [plot_specs[i] if i in plot_specs else np.nan for i in table[['Retractor','Buckle','Speed']].replace(np.nan,'').apply(tuple, axis=1).values]

#%%
#%%
def get_all_features(write_csv=False):
    i_to_t = get_i_to_t(t)
    feature_funs = {'Min_': [get_min],
                    'Max_': [get_max],
                    'Tmin_': [get_argmin,i_to_t],
                    'Tmax_': [get_argmax,i_to_t]} 
    features = pd.concat(chdata.chdata.get_features(feature_funs).values(),axis=1,sort=True)
    features['12_Chest-Pelvis'] = features['Min_12CHST0000Y7ACXC']-features['Min_12PELV0000Y7ACXA']
    features['14_Chest-Pelvis'] = features['Min_14CHST0000Y7ACXC']-features['Min_14PELV0000Y7ACXA']
    features['16_Chest-Pelvis'] = features['Min_16CHST0000Y7ACXC']-features['Min_16PELV0000Y7ACXA']

    if write_csv:
        features.to_csv(directory + 'features.csv')
    return features

features = get_all_features(write_csv=False)
#%% 

# align all data to the peak onset, defined as 10ms before the first time at which the acceleration exceeds 10% of its peak
# booleans are median filtered with a kernel size of 11 to account for noise
medfilt11 = partial(medfilt, kernel_size=11)

onsets = {'12HEAD0000Y7ACXA': 'min',
          '14HEAD0000Y7ACXA': 'min',
          '16HEAD0000Y7ACXA': 'min'}
onset = pd.DataFrame(index=chdata.index)
for j in chdata.columns.drop('S0SLED000000ACXD'):
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
plot_channels = [['12HEAD0000Y7ACXA','14HEAD0000Y7ACXA','16HEAD0000Y7ACXA'],
                 ['12CHST0000Y7ACXC','14CHST0000Y7ACXC','16CHST0000Y7ACXC'],
                 ['12PELV0000Y7ACXA','14PELV0000Y7ACXA','16PELV0000Y7ACXA']]  

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



# Marc retractor + original buckle + in vehicle @ 48
#subset = pd.concat([table.query('Retractor==\'Marc_4\' and Buckle==\'Original\''),
#                    table.query('Retractor==\'Vehicle\' and Buckle==\'Vehicle\' and Speed==48')], axis=0)

# Marc retractor + original buckle + in vehicle + Marc retractor + flexible buckle @ 48
subset = pd.concat([table.query('Retractor==\'Marc_4\' and Buckle==\'Original\''),
                    table.query('Retractor==\'Vehicle\' and Buckle==\'Vehicle\' and Speed==48'),
                    table.query('Retractor==\'Marc_4\' and Buckle==\'Short Flex\'')], axis=0)


# in vehicle vs on the bench 
#subset = pd.concat([table.query('Retractor==\'None\' and Buckle==\'Original\' and Model==\'Turbo Booster\''),
#                    table.query('Retractor==\'Vehicle\' and Buckle==\'Vehicle\' and Model==\'Turbo Booster\' and Speed==48')], axis=0)


# Marc retractor + flexible buckle vs. Marc retractor + original buckle
#subset = table.query('Retractor==\'Marc_4\'')

# Marc retractor + flexible buckle + in vehicle @ 48
#subset = pd.concat([table.query('Retractor==\'Marc_4\' and Buckle==\'Short Flex\''),
#                    table.query('Retractor==\'Vehicle\' and Buckle==\'Vehicle\' and Speed==48')], axis=0)


subset['Retractor'] = subset['Retractor'].replace('Marc_4','Bench\n+ Retractor')
subset['Buckle'] = subset['Buckle'].replace({'Vehicle': '', 'Original': '','Short Flex': '\n(Flexible Buckle)'})
subset['Cond'] = subset[['Retractor','Buckle']].apply(lambda x: ''.join(x), axis=1)



grouped = subset.groupby('Cond')

for ch in plot_channels:
    fig, ax = plt.subplots()
    legend_labels = []
    legend_lines = []
    for grp in grouped:
        legend_labels.append(grp[0])
        
        i_ch = [[j for j in ch if i in j][0] for i in grp[1]['Pos'].values]
        merged_name = get_merged_name(ch)
        
        # merge data into one Series
        x = pd.DataFrame({'ch': np.concatenate([chdata.at[i, j] for i, j in zip(grp[1].index, i_ch)]),
                          't': np.tile(t, len(grp[1])),
                          'Cond': np.repeat(grp[1]['Cond'].values, len(t))})   
#       if n<=3, plot individual tests!         
        if len(grp[1])<=3:
            palette = grp[1][['Cond','Color']].set_index('Cond',drop=True).squeeze()
            if isinstance(palette, tuple):
                palette = {grp[0]: palette}
            else:
                palette = palette.to_dict()
            ax = sns.lineplot(x='t', y='ch', hue='Cond', data=x, ci=None, ax=ax, palette=palette)
            legend_lines.append(ax.lines[-1])
        else:
            palette = grp[1][['Cond','Color']].set_index('Cond',drop=True).squeeze().to_dict()
            ax = sns.lineplot(x='t', y='ch', hue='Cond', data=x, ax=ax, linewidth=0.5, linestyle='--', palette=palette)
            legend_lines.append(ax.lines[-1])
    ax.legend(legend_lines, legend_labels, bbox_to_anchor=(1,-0.2), fontsize=16, ncol=1)
    ax.set_xlim([0, 0.12])
    ax = set_labels(ax, {'title': renameISO(merged_name), 'xlabel': 'Time [s]', 'ylabel': get_units(merged_name)})
    ax = adjust_font_sizes(ax, {'title': 20, 'axlabels': 18, 'ticklabels': 16})

    # figure background colour
    fig.patch.set_facecolor([64/255, 64/255, 64/255])
    
    # axis label colour
    ax.xaxis.label.set_color('#ffffff')
    ax.yaxis.label.set_color('#ffffff')
    ax.tick_params(colors='#ffffff')
    ax.title.set_color('#ffffff')
                       
    # axis face colour
#    ax.set_facecolor([0.96, 0.96, 0.96])
    
#%%
    
subset = (table.table.query_list('Retractor',['Marc_4','Vehicle','None'])
               .table.query_list('Buckle',['Original','Short Flex','Vehicle'])
               .table.query_list('Speed',[np.nan, 48])
               .query('Model==\'Harmony\''))
subset['Cond'] = subset[['Retractor','Buckle']].apply(lambda x: ''.join(x), axis=1)

plot_channels = [['12_Chest-Pelvis','14_Chest-Pelvis','16_Chest-Pelvis']]
for ch in plot_channels:
    i_ch = [[j for j in ch if i in j][0] for i in subset['Pos'].values]
    x = pd.DataFrame({'ch': [features.at[i, j] for i, j in zip(subset.index, i_ch)],
                      'Cond': subset['Cond']})
    fig, ax = plt.subplots()
    ax = sns.barplot(x='Cond', y='ch', data=x)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.axhline(0, linestyle='--', linewidth=1)
#%% plot sled pulse
plot_channels = ['S0SLED000000ACXD',
                 '10CVEHCG0000ACXD']
subset = (table.table.query_list('Retractor',['Marc_4','Vehicle','None'])
               .table.query_list('Buckle',['Short Flex','Original']))
for ch in plot_channels:
    x = {'Sled': chdata[ch]}
    tlen = chdata[ch].apply(len).values[0]
    fig, ax = plt.subplots()
    ax = plot_overlay(ax, t_[:tlen], x, line_specs={'Sled': {'color': 'k', 'linewidth': 1}})
    ax.set_xlim([0, 0.12])
    ax.set_ylim([-12, 60])
    ax = set_labels(ax, {'title': 'Vehicle' if 'VEH' in ch else 'Sled', 'xlabel': 'Time [s]', 'ylabel': get_units(ch)})
    ax = adjust_font_sizes(ax, {'title': 20, 'axlabels': 18, 'ticklabels': 16})