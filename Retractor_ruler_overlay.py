# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 19:38:39 2019
Script to overlay ruler at peak head excursion
@author: tangk
"""

from PIL import Image
import imageio
import glob
import pandas as pd
import numpy as np

directory = 'P:\\2020\\20-5000\\20-5002\\Tests - Retractor\\'

#%%
get_center_overlay = False
get_knee_overlay = False
table = pd.read_excel(directory + '20-5002 Overview.xlsx', dtype={'Number': str}).set_index('Number')

# get overlay
if get_center_overlay:
    overlay_reader = imageio.get_reader(directory + 'Excursion measurements\\Center Ruler.avi')
    overlay_writer = imageio.get_writer(directory + 'center_excursion_ruler.jpg')
    overlay_writer.append_data(overlay_reader.get_data(0))
    overlay_writer.close()

if get_knee_overlay:
    overlay_reader = imageio.get_reader(directory + 'Excursion measurements\\Knee Ruler.avi')
    overlay_writer = imageio.get_writer(directory + 'knee_excursion_ruler.jpg')
    overlay_writer.append_data(overlay_reader.get_data(0))
    overlay_writer.close()

overlay = Image.open(directory + 'knee_excursion_ruler.jpg').convert(mode='RGBA')
overlay.putalpha(90)

# get videos
videos = glob.glob(directory + '*\\Videos\\* Top Belt 8mm.avi')
for v in videos:
    i = table.loc[v[42:45], 'Head Excursion Time'] + table.loc[v[42:45], 't0']
    if np.isnan(i):
        continue
    else:
        i = int(i)
    # write the peak excursion frame
    reader = imageio.get_reader(v)
    writer = imageio.get_writer(directory + v[42:45] + ' SE19-0054_5 Knee Excursion.jpg')
    writer.append_data(reader.get_data(i))
    writer.close()
    
    # read it in pillow and ovelay the ruler
    im = Image.open(directory + v[42:45] + ' SE19-0054_5 Knee Excursion.jpg').convert(mode='RGBA')
    im.alpha_composite(overlay)
    im = im.convert(mode='RGB')
    im.save(directory + v[42:45] + ' Knee Excursion Overlay.jpg')
    im.close()

#%% takata from 2019-5002
v = 'Q:\\2019\\19-5000 incomplet\\19-5002\\Tests - Retractor\\031-SE19-0054_5  TC19-080\\Video\\SE19-0054_5_Left View 6mm.avi'
i = 83
#i = 67

# write the peak excursion frame
reader = imageio.get_reader(v)
writer = imageio.get_writer(directory + v[42:45] + ' SE19-0054_5 Left.jpg')
writer.append_data(reader.get_data(i))
writer.close()
