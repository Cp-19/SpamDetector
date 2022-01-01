# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 14:57:42 2022

@author: DELL
"""

import numpy as np 
import pandas as pd

df=pd.read_csv("spam.csv")
print(df.columns)
print(df['v2'].head())

