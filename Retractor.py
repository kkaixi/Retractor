# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:30:50 2018

@author: tangk
"""
import pandas as pd
import numpy as np
from PMG.read_data import initialize
from PMG.COM.arrange import *
from PMG.COM.plotfuns import *

directory = 'P:\\Data Analysis\\Projects\\Retractor\\'
cutoff = range(100,1600)
channels = ['12CHST0000Y7ACXC',
            '14CHST0000Y7ACXC']
drop = ['SE17-0039_3']

table, t, chdata = initialize(directory,channels, cutoff, drop=drop)

#%% Compare of in-vehicle vs. bench + original buckle + vehicle retractor vs. bench + flexible buckle + vehicle retractor 
plot_channels = [['12CHST0000Y7ACXC','14CHST0000Y7ACXC']]
#subset = (table.query('Model==\'Turbo Booster\'')
#               .table.query_list('Buckle',['Original','Short Flex','Vehicle'])
#               .table.query_list('Retractor', ['Marc_4','None','Vehicle']))


subset = (table.query('Model==\'Harmony\'')
               .table.query_list('Buckle',['Short Flex'])
               .table.query_list('Retractor',['Vehicle','Marc_4'])) 


subset['Cond'] = subset[['Retractor','Buckle']].apply(lambda x: '+'.join(x), axis=1)
grouped = subset.groupby('Cond')
for grp in grouped:
    for ch in plot_channels:
        x = {'a': merge_columns(chdata.loc[grp[1].index, ch])}
        fig, ax = plt.subplots()
        ax = plot_overlay(ax, t, x)
        ax = set_labels(ax, {'title': grp[0]})
        ax.set_ylim([-70, 10])