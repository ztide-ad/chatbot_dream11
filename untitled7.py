# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 01:37:58 2024

@author: anmol
"""
import pandas as pd
from model.train_model import train_model
df=pd.read_csv(r"C:\Users\anmol\prod_features\data\train_dataset.csv")
train_model(df)