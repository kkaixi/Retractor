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


#%%

dataset.get_data(['timeseries','features'])

names = pd.read_csv(directory+'names.csv',index_col=0,header=None,squeeze=True)
names = names.append(pd.Series({ch: renameISO(ch) for ch in dataset.channels}))
rename = partial(rename, names=names)

#%% preprocessing
flip_sign = [i for i in dataset.table.query('Year==2020 and ATD==\'Y7\'').index if i in dataset.timeseries.index]
dataset.timeseries.loc[flip_sign, '12CHST0000Y7ACXC'] = dataset.timeseries.loc[flip_sign, '12CHST0000Y7ACXC'].apply(lambda x: -x)
dataset.timeseries.at['TC57-999_9', '12HEAD0000Y7ACXA'] = dataset.timeseries.at['TC57-999_9', '12HEADRD00Y7ACXA']
dataset.timeseries = dataset.timeseries.applymap(lambda x: x-x[0])
#dataset.features.at['TC57-999_9', 'Min_12HEAD0000Y7ACXA'] = dataset.timeseries.at['TC57-999_9', '12HEAD0000Y7ACXA'].min()
dataset.timeseries.at['SE19-0054_5','S0SLED000000ACXD'] = -dataset.timeseries.at['SE19-0054_5','S0SLED000000ACXD']
#%% plot excursion values--stacked
plot_channels = ['Head_Excursion',
                 'Knee_Excursion']

# Marc vs. original
#subset = dataset.table.query('Config==\'sled_213_new_decel\' and Pulse==\'213\'')

# Marc vs. vehicle
subset = dataset.table.query('Config==\'sled_213_new_decel\' and Model==\'Bubble Bum\' and Retractor!=\'None\'')
subset['cond'] = subset[['Pulse','Retractor','Buckle']].apply(lambda x: ' '.join(x), axis=1)



subset['Model'] = subset['Model'].apply(rename)
grouped = subset.groupby(['ATD'])

for ch in plot_channels:
    for grp in grouped:
        sgrouped = grp[1].groupby(['Retractor'])
        fig, ax = plt.subplots()
        
        
#        ax = sns.barplot(x='Model', y=ch, data=sgrouped.get_group('Marc_4'), ci='sd', capsize=0.15, color='#c2c2c2', order=['A','B','C'], ax=ax)
#        ax = sns.barplot(x='Model', y=ch, data=sgrouped.get_group('None'), ci='sd', capsize=0.15, color='#5c5c5c', order=['A','B','C'], ax=ax)
#        print(grp[0])
#        print(sgrouped.get_group('Marc_4').groupby('Model').std()[['Head_Excursion','Knee_Excursion']])
        
        ax = sns.barplot(x='cond', y=ch, order=['TC17-110_88pct Caravan short_flex', '213 Marc_4 short_flex',
                                                'TC17-110_88pct Marc_4 short_flex', 'TC17-110_88pct Marc_4 original'],
                         data=grp[1], ci='sd', capsize=0.15)
        plt.xticks(rotation=90)
        print(grp[0])
        print(grp[1][['Head_Excursion','Knee_Excursion']].agg(['max','min']))
        
        
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


subset = dataset.table.query('Pulse==\'213\'')
subset['cond'] = subset[['Pulse','Retractor','Buckle']].apply(lambda x: ' '.join(x), axis=1)

grouped = subset.groupby(['ATD','Model', 'cond'])
for grp in grouped:
    fig, ax = plt.subplots()
#        x = arrange_by_group(subset, dataset.timeseries.loc[grp[1].index, ch], 'cond')
    
    x = arrange_by_group(subset, dataset.timeseries.loc[grp[1].index, '12CHST0000{}ACXC'.format(grp[0][0])], 'cond')
    x['chest'] = x.pop(grp[0][2])
    ax = plot_overlay(ax, dataset.t, x)
    y = arrange_by_group(subset, dataset.timeseries.loc[grp[1].index, '12PELV0000{}ACXA'.format(grp[0][0])], 'cond')
    y['pelvis'] = y.pop(grp[0][2])
    ax = plot_overlay(ax, dataset.t, y)
    ax.set_ylim([-80, 10])
    
    ax.set_title(grp[0])
    ax.legend(bbox_to_anchor=(1,1))
    
    print(grp[0])
    print('chest')
    print(dataset.t[int(x['chest'].apply(np.argmin).mean())])
    print('pelvis')
    print(dataset.t[int(y['pelvis'].apply(np.argmin).mean())])
    print('\n')
        
#%% plot each channel with different conditions in different colours
plot_channels = ['HEAD0000Y7ACXA',
                 'CHST0000Y7ACXC',
                 'PELV0000Y7ACXA',
                 'HEAD0000Q6ACXA',
                 'CHST0000Q6ACXC',
                 'PELV0000Q6ACXA']
#plot_channels = ['S0SLED000000ACXD']
#plot_channels = dataset.timeseries.filter(like='Y7').columns

subset = dataset.table.query('Pulse==\'213\' and Retractor==\'Marc_4\'')
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
        ax = plot_overlay(ax, dataset.t, x, line_specs={key: {'color': '#000000'} for key in x.keys()})
        ax = set_labels(ax, {'title': '{0} ({1})'.format(renameISO(get_merged_name(ch_pos)), rename(grp[0][0])),
                             'xlabel': 'Time [s]',
                             'ylabel': get_units(get_merged_name(ch_pos))})
        ax = adjust_font_sizes(ax, {'ticklabels': 18,
                                    'title': 20,
                                    'axlabels': 18})
        ax.set_ylim([-70, 10])        
#        ax.legend(bbox_to_anchor=(1,1))
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

# comparing Marc retractor vs. no retractor
subset = dataset.table.query('Pulse==\'213\'')
subset['cond'] = subset[['Config', 'Pulse', 'Retractor', 'Buckle']].replace(np.nan, '').apply(lambda x: ' '.join(x), axis=1)

# comparing Marc retractor vs. vehicle retractor
#subset = dataset.table.query('Config==\'sled_213_new_decel\' and Retractor!=\'None\' and Model==\'Bubble Bum\'')
#subset['cond'] = subset[['Pulse', 'Retractor', 'Buckle']].replace(np.nan, '').apply(lambda x: ' '.join(x), axis=1)

subset['Model'] = subset['Model'].apply(rename)
subset['cond'] = subset['cond'].apply(rename)
grouped = subset.groupby(['ATD'])

for grp in grouped:
    for ch in plot_channels:
        data = merge_columns(dataset.features.loc[grp[1].index, ch])
        merged_name = get_merged_name(ch)
        
        if 'Pelvis-Chest' in merged_name:
            title = 'Pelvis-Chest ({})'.format(rename(grp[0]))
            ylim = [-15, 15]
        elif merged_name.startswith('Tmin'):
            title = 'Time to Peak ' + renameISO(merged_name[5:]) + ' ({})'.format(rename(grp[0]))
            ylim = [0, 0.1]
        elif merged_name.startswith('Min'):
            title = 'Peak ' + renameISO(merged_name[4:]) + ' ({})'.format(rename(grp[0]))
            ylim = [0, 80]
            data = -data

        data = pd.concat((grp[1], data), axis=1)
        fig, ax = plt.subplots()
        

            
        ax = sns.barplot(x='Model', y=merged_name, hue='cond', ci='sd', capsize=0.15, order=['A','B','C'], 
                         palette={'Modified Belt Assembly': '#c2c2c2',
                                  'Original Belt Assembly': '#5c5c5c'}, 
                         hue_order=['Original Belt Assembly','Modified Belt Assembly'], ax=ax, data=data)
        ax.get_legend().remove()
        
#        ax = sns.barplot(x='cond', y=merged_name, ci='sd', capsize=0.15, 
#                         order = ['TC17-110_88pct Caravan short_flex', '213 Marc_4 short_flex',
#                                  'TC17-110_88pct Marc_4 short_flex', 'TC17-110_88pct Marc_4 original'],
#                         ax=ax, data=data)
#        ax.set_xticklabels([])
        


            
            
            
            
        ax = set_labels(ax, {'title': title, 'xlabel': 'Model', 'ylabel': get_units(merged_name)})
        ax = adjust_font_sizes(ax, {'title': 24,
                                    'xlabel': 24,
                                    'ylabel': 18,
                                    'xticklabel': 24,
                                    'yticklabel': 18})  
        ax.set_ylim(ylim)
        if 'Pelvis-Chest' in merged_name:
            ax = set_labels(ax, {'ylabel': 'Acceleration [g]'})
            
#        values = data.groupby('cond').mean()[merged_name]
#        veh_ctrl = values['TC17-110_88pct Caravan short_flex']
#        print(grp[0])
##        print(values)
#        print(values.apply(lambda x: x-veh_ctrl))
#        print('\n')
#        print(grp[0])
#        print(data.groupby(['Model','cond']).mean()[merged_name].groupby(level=0).diff())
#        print('\n')
        
        print(grp[0])
        print(data[['Model','cond',merged_name]].groupby(['cond','Model']).agg(['mean','std']))

#%% plot peaks in vehicles 
plot_channels = ['Min_xxHEAD0000Y7ACXA',
                 'Min_xxCHST0000Y7ACXC',
                 'Min_xxPELV0000Y7ACXA',
                 'Min_xxHEAD0000Q6ACXA',
                 'Min_xxCHST0000Q6ACXC',
                 'Min_xxPELV0000Q6ACXA',
                 'Y7_Pelvis-Chest',
                 'Q6_Pelvis-Chest']

subset = dataset.table.query('(Pulse==\'213\' and Retractor==\'Marc_4\') or Config==\'ffrb\'')

grouped = subset.groupby(['ATD'])
for grp in grouped:
    for ch in plot_channels:
        if grp[0] not in ch:
            continue
        ch_pos = [ch.replace('xx',j) for i in grp[1].index.unique() for j in grp[1].loc[[i], ['Pos']].astype(str).values.flatten()]
        data = pd.Series([dataset.features.at[grp[1].index[i], ch_pos[i]] for i in range(len(ch_pos))], index=grp[1].index)
        
        if len(data)==0:
            continue
        
        if 'min' in ch.lower(): 
            data = -data
        
        if data.index.duplicated().any():
            data = data.reset_index(drop=True)
            x = grp[1].reset_index(drop=True)
        else:
            x = grp[1]
        x = pd.concat((x, data.to_frame(name=ch)), axis=1)
            
        if len(x)==0: 
            continue
        fig, ax = plt.subplots()
        ax = sns.barplot(x='Model', y=ch, hue='Config', ax=ax, data=x)
        ax = adjust_font_sizes(ax, {'ticklabels': 18,
                                    'title': 20,
                                    'axlabels': 18})
        ax.legend(bbox_to_anchor=(1,1))
        plt.show()
        plt.close(fig)

#%% plot sled pulse
ch = 'S0SLED000000ACXD'
subset = dataset.table.query('Pulse==\'213\' or Pulse==\'TC17-110_88pct\'')
grouped = subset.groupby('Pulse')

for grp in grouped:
    fig, ax = plt.subplots()
    x = {'x': dataset.timeseries.loc[grp[1].index, ch]}
    ax = plot_overlay(ax, dataset.t, x, line_specs = {'x': {'color': 'k'}})
    ax = set_labels(ax, {'title': rename(grp[0]), 'xlabel': 'Time [s]', 'ylabel': get_units(ch)})
    ax = adjust_font_sizes(ax, {'title': 20, 'ticklabels': 18, 'axlabels': 18})
    ax.set_ylim([-30, 10])
    print(x['x'].apply(np.min).agg(['max','min']))

#%%
import xlwings as xw
import xlsxwriter

def plot_excel(ax, workbook=None):
    if workbook is None:
        wb = xw.Book()
        sht = wb.sheets['Sheet1']
    
    # get axis labels
    ax_labels = {'title': ax.title.get_text(),
                 'xlabel': ax.xaxis.get_label().get_text(),
                 'ylabel': ax.yaxis.get_label().get_text(),
                 'legend': ax.get_legend()}
    
    # print the values
    
    
