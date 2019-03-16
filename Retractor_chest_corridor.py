# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 07:29:27 2019

Chest response corridor for retractor

@author: tangk
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PMG.read_data import initialize, get_test_info
from PMG.COM.arrange import unpack
from PMG.COM.plotfuns import *

def read_data():
    tc = ['TC13-130',
          'TC13-133',
          'TC13-207',
          'TC16-111',
          'TC16-113',
          'TC16-104',
          'TC16-104',
          'TC12-212',
          'TC17-113',
          'TC15-111',
          'TC14-016',
          'TC12-004',
          'TC17-205',
          'TC16-127',
          'TC18-110',
          'TC16-129',
          'TC18-105',
          'TC17-207',
          'TC16-140']
    
    channels = ['14CHST0000Y7ACXC',
                '16CHST0000Y7ACXC']
    
    tests, _ = get_test_info()
    tc = [i for i in tc if i in tests]
    
    _, t, chdata = initialize('P:\\', channels, cutoff, tc=tc)
    return t, chdata
    
def preprocess():
    rm = [['TC16-111', 'TC12-004','TC17-205','TC16-127','TC16-129','TC18-105'], ['16CHST0000Y7ACXC']*6]
    for pair in zip(*rm):
        chdata.at[pair[0], pair[1]] = np.tile(np.nan, len(cutoff))
    return chdata

def get_bands(t):
    x = unpack(chdata.unstack()).unstack().rename('x').to_frame().reset_index(drop=True)
    x['t'] = np.tile(t, chdata.shape[0]*chdata.shape[1])
    fig, ax = plt.subplots()
    ax = sns.lineplot(x='t', y='x', data=x)
    return ax

cutoff = range(100, 1600)
t, chdata = read_data()
chdata = preprocess()



data = pd.read_excel('P:\\2019\\19-5000\\19-5002\\Tests - Retractor\\May 2018\\044-SE18-0053_24  TC19-044\\PMG\\SE18-0053_24(SAI).xls', 
                     header=0, index_col=0, skiprows=[1,2])
data = data.query('T_10000_0>={0} and T_10000_0<={1}'.format(t[0], t[-1]))
t2 = data['T_10000_0']
ax = get_bands(t)
ax.plot(t2, data['12CHST0000Y7ACXC'])
    