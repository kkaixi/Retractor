# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:16:08 2019

@author: tangk
"""

from PMG.read_data import PMGDataset
import numpy as np
import pandas as pd
from PMG.COM.get_props import get_peaks, get_argmax, get_argmin

directory = 'P:\\Data Analysis\\Projects\\Retractor\\'
cutoff = range(100, 1300)
channels = [ 'S0SLED000000ACXD',
             '12HEAD0000Y7ACXA','12HEAD0000Y7ACYA','12HEAD0000Y7ACZA','12HEADRD00Y7ACXA',
             '12NECKUP00Y7FOXA','12NECKUP00Y7FOYA','12NECKUP00Y7FOZA',
             '12NECKUP00Y7MOXB','12NECKUP00Y7MOYB','12NECKUP00Y7MOZB',
             '12NECKLO00Y7FOXA','12NECKLO00Y7FOYA','12NECKLO00Y7FOZA',
             '12NECKLO00Y7MOXB','12NECKLO00Y7MOYB','12NECKLO00Y7MOZB',
             '12CHST0000Y7ACXC','12CHST0000Y7ACYC','12CHST0000Y7ACZC',
             '12CHST0000Y7AVYC',
             '12CHSTRD00Y7ACXC',
             '12CHST0000Y7DSXB',
             '12LUSP0000Y7FOXA','12LUSP0000Y7FOYA','12LUSP0000Y7FOZA',
             '12LUSP0000Y7MOXA','12LUSP0000Y7MOYA','12LUSP0000Y7MOZA',
             '12PELV0000Y7ACXA','12PELV0000Y7ACYA','12PELV0000Y7ACZA',
             '12ILACRILOY7FOXB','12ILACRIUPY7FOXB',
             '12SEBE0000B3FO0D','12SEBE0000B6FO0D',
             '12HEAD0000Q6ACXA','12HEAD0000Q6ACYA','12HEAD0000Q6ACZA',
             '12NECKUP00Q6FOXA','12NECKUP00Q6FOYA','12NECKUP00Q6FOZA',
             '12NECKUP00Q6MOXB','12NECKUP00Q6MOYB','12NECKUP00Q6MOZB',
             '12NECKLO00Q6FOXA','12NECKLO00Q6FOYA','12NECKLO00Q6FOZA',
             '12NECKLO00Q6MOXB','12NECKLO00Q6MOYB','12NECKLO00Q6MOZB',
             '12CHST0000Q6ACXC','12CHST0000Q6ACYC','12CHST0000Q6ACZC',
             '12CHST0000Q6DSXB',
             '12LUSP0000Q6FOXA','12LUSP0000Q6FOYA','12LUSP0000Q6FOZA',
             '12LUSP0000Q6MOXA','12LUSP0000Q6MOYA','12LUSP0000Q6MOZA',
             '12PELV0000Q6ACXA','12PELV0000Q6ACYA','12PELV0000Q6ACZA']
channels = channels + [x.replace('12','14') for x in channels if x.startswith('12')] + [x.replace('12','16') for x in channels if x.startswith('12')]

table_filters = {'query_list': [['Model', ['Takata','Harmony','Bubble Bum']],
                                ['ATD', ['Y7','Q6']],
                                ['Retractor',[np.nan, 'Marc_4', 'None', 'Caravan']],
                                ['Adjust', [3.75, np.nan, 1, 2]],
                                ['Buckle', ['short_flex', np.nan, 'original']]],
                'drop': ['TC57-999_18', 'TC57-999_25', ],
                'query': 'Year>=2019 or Config==\'ffrb\''}

preprocessing = None
    
dataset = PMGDataset(directory, channels=channels, cutoff=cutoff, verbose=False)
dataset.table_filters = table_filters
dataset.preprocessing = preprocessing
#%% read data 


if __name__=='__main__': 
    # if running the script, get the data
    dataset.get_data(['timeseries'])
    dataset.timeseries.at['TC57-999_9', '12HEAD0000Y7ACXA'] = dataset.timeseries.at['TC57-999_9', '12HEADRD00Y7ACXA']
    flip_sign = [i for i in dataset.table.query('Year==2020 and ATD==\'Y7\'').index if i in dataset.timeseries.index]
    dataset.timeseries.loc[flip_sign, '12CHST0000Y7ACXC'] = dataset.timeseries.loc[flip_sign, '12CHST0000Y7ACXC'].apply(lambda x: -x)

    tmins = dataset.timeseries.applymap(get_argmax).rename(lambda x: 'Tmax_' + x, axis=1)
    tmaxs = dataset.timeseries.applymap(get_argmin).rename(lambda x: 'Tmin_' + x, axis=1)
    
    tmins = tmins.applymap(lambda x: dataset.t[int(x)] if not np.isnan(x) else np.nan)
    tmaxs = tmaxs.applymap(lambda x: dataset.t[int(x)] if not np.isnan(x) else np.nan)
    
    table = dataset.table
    features = pd.concat((get_peaks(dataset.timeseries),
                          tmins,
                          tmaxs), axis=1)
    features['Y7_Pelvis-Chest'] = features['Min_12PELV0000Y7ACXA'].abs()-features['Min_12CHST0000Y7ACXC'].abs()
    features['Q6_Pelvis-Chest'] = features['Min_12PELV0000Q6ACXA'].abs()-features['Min_12CHST0000Q6ACXC'].abs()
    features.to_csv(directory + 'features.csv')

    names = {'Bubble Bum': 'A',
             'Harmony': 'B',
             'Takata': 'C',
             'Y7': 'Hybrid III',
             'Q6': 'Q6',
             'Head_Excursion': 'Head Excursion',
             'Knee_Excursion': 'Knee Excursion',
             'sled_213_new_decel 213 Marc_4 short_flex': 'Modified Belt Assembly',
             'sled_213_new_decel 213 None original': 'Original Belt Assembly',
             '213': 'F/CMVSS 213 Pulse',
             'TC17-110_88pct': 'Vehicle Pulse'}
    names = pd.Series(names)
    names.to_csv(directory + 'names.csv')