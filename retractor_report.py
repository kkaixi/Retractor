# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 16:41:30 2018

@author: tangk
"""
import matplotlib.pyplot as plt

c1 = np.array([3,0,30])/255
c2 = np.array([115,3,192])/255
c3 = np.array([236,56,188])/255
colours = [c1,c2,c3]

def six_panel(t,chdata,table,fsize=(10,10),colours=[[0,0,0],[0.3,0.3,0.3],[0.6,0.6,0.6]]):
    f,axs = plt.subplots(nrows=3,ncols=2,figsize=fsize)
    channels = chdata.columns
    for i, ax in enumerate(axs.flatten()):
        if i-1 > len(channels):
            break
        
        ch = channels[i]
        
        for j in chdata.index:
            if table['Retractor'][j]=='Vehicle':
                ax.plot(t,chdata[ch][j],color=colours[0])
            elif table['Retractor'][j]=='UMTRI':
                ax.plot(t,chdata[ch][j],color=colours[1])
            else:
                ax.plot(t,chdata[ch][j],color=colours[2])
        
        if ch=='12HEAD0000Y7ACXA':
            ax.set_title('Head Acceleration')
            ax.set_ylabel('Acceleration [g]')
        elif ch=='12CHST0000Y7ACXC':
            ax.set_title('Chest Acceleration')
            ax.set_ylabel('Acceleration [g]')
        elif ch=='12PELV0000Y7ACXA':
            ax.set_title('Pelvis Acceleration')
            ax.set_ylabel('Acceleration [g]')
        elif ch=='12SEBE0000B3FO0D':
            ax.set_title('Lap Belt Load')
            ax.set_ylabel('Load [N]')
        elif ch=='12SEBE0000B6FO0D':
            ax.set_title('Shoulder Belt Load')
            ax.set_ylabel('Acceleration [g]')
        else:
            ax.set_title(ch)
            if 'AC' in ch:
                ax.set_ylabel('Acceleration [g]')
        ax.set_xlabel('Time [s]')
    
six_panel(t,chdata,table)