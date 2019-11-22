# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:48:25 2019

Code for retractor (Nov 2019)

@author: tangk
"""

##from PMG.COM.mme import MMEData
#from PMG.COM.get_props import get_peaks
from PMG.COM.arrange import arrange_by_group
from PMG.COM.plotfuns import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from initialize import dataset
from PMG.COM.arrange import get_merged_name, merge_columns
from functools import partial 
from PMG.COM.easyname import get_units, rename, rename_list, renameISO


#%% workarounds for reading mme stuff
#real_table = dataset.table
#dataset.table = dataset.table.query('Year<2020')
#dataset = dataset.read_timeseries()
#
#all_data = {i: unpack(dataset.timeseries.loc[i]) for i in dataset.table.index}
#
#dataset.table = real_table
#mme_paths = glob.glob('P:\\2020\\20-5000\\20-5002\\Tests - Retractor\\**\\SE*.mme', recursive=True)
#
#for path in mme_paths:
#    mme = MMEData(path)
#    se = mme.info['Laboratory test ref. number']
#    if se not in dataset.table.index:
#        print('skip ' + se)
#        continue
#    all_data[se] = mme.to_dataframe().loc[400:].reset_index(drop=True)
#dataset.timeseries = to_chdata(all_data, cutoff=cutoff)
#dataset.t = dataset.t[cutoff]

dataset.get_data(['timeseries','features'])

names = pd.read_csv(directory+'names.csv',index_col=0,header=None,squeeze=True)
names = names.append(pd.Series({ch: renameISO(ch) for ch in dataset.channels}))
rename = partial(rename, names=names)
#%%
#plot_specs = {'Bubble Bum': {'color': },
#              'Harmony':    {'color': },
#              'Takata':     {'color': '#589F58'}}
#sns_plot_specs = {'palette': {}}
#
#dataset.table['cond'] = dataset.table[['Config', 'Pulse', 'Model', 'Retractor', 'Buckle']].replace(np.nan, '').apply(lambda x: ' '.join(x), axis=1)
#plot_specs = dict.fromkeys(dataset.table['cond'].unique())
#
#plot_specs['sled_213_new_decel 213 Takata Marc_4 short_flex'] =     {'color': '#56c77d'}
#plot_specs['sled_213_new_decel 213 Takata None original'] =         {'color': '#176331'}
#plot_specs['sled_213_new_decel 213 Harmony Marc_4 short_flex'] =    {'color': '#ebb0f7'} 
#plot_specs['sled_213_new_decel 213 Harmony None original'] =        {'color': '#8f21a6'}
#plot_specs['sled_213_new_decel 213 Bubble Bum Marc_4 short_flex'] = {'color': '#d1d1d1'} 
#plot_specs['sled_213_new_decel 213 Bubble Bum None original'] =     {'color': '#636363'}
#
#plot_specs['sled_213_new_decel TC17-110_88pct Bubble Bum Caravan short_flex'] = None
#plot_specs['sled_213_new_decel TC17-110_88pct Bubble Bum Marc_4 short_flex'] = None
#plot_specs['sled_213_new_decel TC17-110_88pct Bubble Bum Marc_4 original'] = None
#plot_specs['ffrb  Harmony  '] = None
#plot_specs['ffrb  Bubble Bum  '] = None
#plot_specs['ffrb  Takata  '] = None
#
#
#
#sns_specs = {'palette': {'sled_213_new_decel 213 Takata Marc_4 short_flex':     '#56c77d',
#                         'sled_213_new_decel 213 Takata None original':         '#176331',
#                         'sled_213_new_decel 213 Harmony Marc_4 short_flex':    '#ebb0f7',
#                         'sled_213_new_decel 213 Harmony None original':        '#8f21a6',
#                         'sled_213_new_decel 213 Bubble Bum Marc_4 short_flex': '#d1d1d1',
#                         'sled_213_new_decel 213 Bubble Bum None original':     '#636363'}}

#%% preprocessing
flip_sign = [i for i in dataset.table.query('Year==2020 and ATD==\'Y7\'').index if i in dataset.timeseries.index]
dataset.timeseries.loc[flip_sign, '12CHST0000Y7ACXC'] = dataset.timeseries.loc[flip_sign, '12CHST0000Y7ACXC'].apply(lambda x: -x)
dataset.timeseries.at['TC57-999_9', '12HEAD0000Y7ACXA'] = dataset.timeseries.at['TC57-999_9', '12HEADRD00Y7ACXA']
dataset.timeseries = dataset.timeseries.applymap(lambda x: x-x[0])
#dataset.features.at['TC57-999_9', 'Min_12HEAD0000Y7ACXA'] = dataset.timeseries.at['TC57-999_9', '12HEAD0000Y7ACXA'].min()

#%% plot excursion values--stacked
plot_channels = ['Head_Excursion',
                 'Knee_Excursion']

subset = dataset.table.query('Config==\'sled_213_new_decel\' and Pulse==\'213\'')
subset['Model'] = subset['Model'].apply(rename)
grouped = subset.groupby(['ATD'])

for ch in plot_channels:
    for grp in grouped:
        sgrouped = grp[1].groupby(['Retractor'])
        fig, ax = plt.subplots()
        ax = sns.barplot(x='Model', y=ch, data=sgrouped.get_group('Marc_4'), ci='sd', capsize=0.15, color='#c2c2c2', order=['A','B','C'], ax=ax)
        ax = sns.barplot(x='Model', y=ch, data=sgrouped.get_group('None'), ci='sd', capsize=0.15, color='#5c5c5c', order=['A','B','C'], ax=ax)
        if 'Head' in ch:
            ax.set_ylim((300, 550))
        elif 'Knee' in ch:
            ax.set_ylim((500, 700))
        ax = set_labels(ax, {'title': ' '.join((rename(grp[0]), rename(ch))), 
                             'xlabel': 'Model', 
                             'ylabel': get_units(ch)})
        ax = adjust_font_sizes(ax, {'ticklabels': 18,
                                    'title': 20,
                                    'axlabels': 18})      
#%% plot chest and pelvis on the same set of axes
plot_channels = ['12CHST0000Y7ACXC',
                 '12PELV0000Y7ACXA']

subset = dataset.table.query('ATD==\'Y7\' and Pulse==\'213\'')
subset['cond'] = subset[['Pulse','Retractor','Buckle']].apply(lambda x: ' '.join(x), axis=1)

grouped = subset.groupby(['Model', 'cond'])
for grp in grouped:
    for ch in plot_channels:
        fig, ax = plt.subplots()
#        x = arrange_by_group(subset, dataset.timeseries.loc[grp[1].index, ch], 'cond')
        
        x = arrange_by_group(subset, dataset.timeseries.loc[grp[1].index, '12CHST0000Y7ACXC'], 'cond')
        x['chest'] = x.pop(grp[0][1])
        ax = plot_overlay(ax, dataset.t, x)
        x = arrange_by_group(subset, dataset.timeseries.loc[grp[1].index, '12PELV0000Y7ACXA'], 'cond')
        x['pelvis'] = x.pop(grp[0][1])
        ax = plot_overlay(ax, dataset.t, x)
        ax.set_ylim([-80, 10])
        
        ax.set_title((ch, grp[0]))
        ax.legend(bbox_to_anchor=(1,1))
        
#%% plot each channel with different conditions in different colours
plot_channels = ['HEAD0000Y7ACXA',
                 'CHST0000Y7ACXC',
                 'PELV0000Y7ACXA',
                 'HEAD0000Q6ACXA',
                 'CHST0000Q6ACXC',
                 'PELV0000Q6ACXA']
#plot_channels = ['S0SLED000000ACXD']
#plot_channels = dataset.timeseries.filter(like='Y7').columns

subset = dataset.table.query('Pulse==\'213\' or Config==\'ffrb\'')
subset['cond'] = subset[['Config', 'Pulse', 'Retractor', 'Buckle']].replace(np.nan, '').apply(lambda x: ' '.join(x), axis=1)

grouped = subset.groupby(['ATD', 'Model'])
for grp in grouped:
    for ch in plot_channels:
        
        ch_pos = [j + ch for i in grp[1].index.unique() for j in grp[1].loc[[i], ['Pos']].astype(str).values.flatten()]
        data = pd.Series([dataset.timeseries.at[grp[1].index[i], ch_pos[i]] for i in range(len(ch_pos))], index=grp[1].index)
        
#        pd.Series([dataset.timeseries.at[i, grp[1].loc[i, 'Pos'].astype(str) + ch] for i in grp[1].index], index=grp[1].index)
        
#        data = merge_columns(dataset.timeseries.loc[grp[1].index, ch])
        if len(data)==0:
            continue
        
        if data.index.duplicated().any():
            data = data.reset_index(drop=True)
            table = grp[1].reset_index(drop=True)
            x = arrange_by_group(table, data, 'cond')
        else:
            x = arrange_by_group(grp[1], data, 'cond')
            
        if len(x)==0: 
            continue
        fig, ax = plt.subplots()
        ax = plot_overlay(ax, dataset.t, x)
        
        ax.set_title((get_merged_name(ch), grp[0]))
        ax.legend(bbox_to_anchor=(1,1))
        plt.show()
        plt.close(fig)

#%% print range of values in triplicates
plot_channels = ['12HEAD0000Y7ACXA',
                 '12CHST0000Y7ACXC',
                 '12PELV0000Y7ACXA',
                 '12HEAD0000Q6ACXA',
                 '12CHST0000Q6ACXC',
                 '12PELV0000Q6ACXA']
subset = dataset.table.query('Pulse==\'213\' and Retractor==\'Marc_4\' and Model==\'Bubble Bum\'')
grouped = subset.groupby(['ATD'])
for grp in grouped:
    print(grp[0])
    print(dataset.timeseries.loc[grp[1].index, plot_channels].applymap(np.nanmin).apply(lambda x: x.max()-x.min()))
    print(dataset.timeseries.loc[grp[1].index, plot_channels].applymap(lambda x: np.nanargmin(x) if not is_all_nan(x) else np.nan).apply(lambda x: x.max()-x.min()))

#%% plot differences in peaks
plot_channels = [['Min_12HEAD0000Y7ACXA','Min_12HEAD0000Q6ACXA'],
                 ['Min_12CHST0000Y7ACXC','Min_12CHST0000Q6ACXC'],
                 ['Min_12PELV0000Y7ACXA','Min_12PELV0000Q6ACXA'],
                 ['Tmin_12HEAD0000Y7ACXA','Tmin_12HEAD0000Q6ACXA'],
                 ['Tmin_12CHST0000Y7ACXC','Tmin_12CHST0000Q6ACXC'],
                 ['Tmin_12PELV0000Y7ACXA','Tmin_12PELV0000Q6ACXA'],
                 ['Y7_Pelvis-Chest','Q6_Pelvis-Chest']]

subset = dataset.table.query('Pulse==\'213\'')
subset['Model'] = subset['Model'].apply(rename)
subset['cond'] = subset[['Config', 'Pulse', 'Retractor', 'Buckle']].replace(np.nan, '').apply(lambda x: ' '.join(x), axis=1)
subset['cond'] = subset['cond'].apply(rename)
grouped = subset.groupby(['ATD'])

for grp in grouped:
    for ch in plot_channels:
        data = merge_columns(dataset.features.loc[grp[1].index, ch])
        merged_name = get_merged_name(ch)
        if (data.dropna()<0).all():
            data = -data
        data = pd.concat((grp[1], data), axis=1)
        fig, ax = plt.subplots()
        ax = sns.barplot(x='Model', y=merged_name, hue='cond', ci='sd', capsize=0.15, order=['A','B','C'], 
                         palette={'Modified Belt Assembly': '#c2c2c2',
                                  'Original Belt Assembly': '#5c5c5c'}, 
                         hue_order=['Original Belt Assembly','Modified Belt Assembly'], ax=ax, data=data)
#        plt.legend(bbox_to_anchor=(1,1))
        ax.get_legend().remove()
        if 'Pelvis-Chest' in merged_name:
            title = 'Pelvis-Chest ({})'.format(rename(grp[0]))
            ylim = [-10, 15]
        elif merged_name.startswith('Tmin'):
            title = 'Time to Peak ' + renameISO(merged_name[5:]) + ' ({})'.format(rename(grp[0]))
            ylim = [0, 0.1]
        elif merged_name.startswith('Min'):
            title = 'Peak ' + renameISO(merged_name[4:]) + ' ({})'.format(rename(grp[0]))
            ylim = [0, 80]
        ax = set_labels(ax, {'title': title, 'xlabel': 'Model', 'ylabel': get_units(merged_name)})
        ax = adjust_font_sizes(ax, {'ticklabels': 18,
                                    'title': 20,
                                    'axlabels': 18})  
        ax.set_ylim(ylim)
        if 'Pelvis-Chest' in merged_name:
            ax = set_labels(ax, {'ylabel': 'Acceleration [g]'})

    
#    print(data.groupby(['cond_2','cond_1']).mean()[get_merged_name(ch)])
#    print(data.groupby(['cond_2','cond_1']).mean()[get_merged_name(ch)].groupby(level=0).diff())
