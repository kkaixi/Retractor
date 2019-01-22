# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:30:50 2018

@author: tangk
"""
from PMG.COM.data import import_data
import pandas as pd
from PMG.COM import arrange
directory = 'C:\\Users\\tangk\\Desktop\\Retractor Data\\'
channels = ['12CHST0000Y7ACXC',
            '12CHST0000Y7DSXB',
            '12PELV0000Y7ACXA']
cutoff = range(100,1600)
t, fulldata = import_data(directory,channels)
chdata = arrange.test_ch_from_chdict(fulldata,cutoff)
t = t.get_values()[cutoff]
se = list(chdata.index)

for ch in channels:
    for s in se:
        if s=='SE18-0053_21' or s=='SE18-0053_22':
            plt.plot(t,chdata[ch][s],'b',label='3.75')
        else:
            plt.plot(t,chdata[ch][s],'r',label='4.75')
    plt.title(ch)
    plt.legend()
    plt.show()