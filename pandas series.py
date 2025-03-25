# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 04:58:27 2025

@author: Talong PC
"""
import pandas as pd;
a = [1, 7, 2];

myvar = pd.Series(a);
myvar = pd.Series(a, index = ["x", "y", "z"])
print(myvar);

print(myvar["x"]);

print(myvar[0]); #depricated