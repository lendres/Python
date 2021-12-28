# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""

# Use this to import from another directory.
import seaborn as sns
import matplotlib.pyplot as plt

import lendres.Data
import lendres.Console
import lendres.Plotting

data = lendres.Data.LoadAndInspectData("data.csv")

lendres.Plotting.FormatPlot()
axis = plt.gca()
sns.histplot(data["bmi"], kde=False, ax=axis, palette="winter")
axis.set(title="Test Plot", xlabel="BMI", ylabel="Count")
