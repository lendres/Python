# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""

import lendres


data = lendres.Data.LoadAndInspectData("data.csv")

print("\n\n\n")
data = lendres.Data.LoadAndInspectData("datawitherrors.csv")