# -*- coding: utf-8 -*-
"""
Created on Jan 2023

A dash app which uses ZemaxGlass.py to open .agf glass catalogs and "opticalglass" to open glass vendors' Excel files
# to plot glass maps of optical glasses and infrared materials.

Used python packages: 
    
opticalglass: https://opticalglass.readthedocs.io/en/latest/README.html

ZemaxGlass:   https://github.com/nzhagen/zemaxglass


@author: Jonas Herbst, jonas.herbst@silloptics.de, jonasherbst@gmx.de

# cd C:\Users\Jonas\opticalglass
# git add ./*
# git commit -m ""
# git push

"""
# Make the dash app executable from a .exe
# https://python.plainenglish.io/how-to-convert-your-dash-app-into-an-executable-gui-b1d4271a8fa7


# do the dash glass map plot

# import ZemaxGlass.py
import ZemaxGlass as zg


# https://opticalglass.readthedocs.io/en/latest/OG_Quickstart/OG_Quickstart.html
import opticalglass as og
import opticalglass.glassmap as gm
# define new plotly class in glassmap_plotly
from opticalglass.glassfactory import create_glass


import os
import numpy as np
import pandas as pd


# read new glass catalogue
import opticalglass.optical_glass_agf_class as agf
#import optical_glass_agf_class as agf
catalog = agf.Zemax_infrared_Catalog()

# some tests
filename = r'C:\Users\Jonas\opticalglass\src\opticalglass\data\infrared_2022.xlsx'    
df = og.glass.xl2df(fname = filename)




#plt.close("all")
#bk7 = create_glass('N-BK7', 'Schott')
#print(bk7)

