# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""

# Use this to import from another directory.
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import lendres


# Import some example data to work with.
data = pd.read_csv("data.csv")

lendres.Plotting.FormatPlot()
axis = plt.gca()
sns.histplot(data["bmi"], kde=True, ax=axis, palette="winter")
axis.set(title="Test Plot", xlabel="BMI", ylabel="Count")

figure  = plt.gcf()

# Make sure directory doesn't exist.
lendres.Plotting.DeleteOutputDirectory()

lendres.Plotting.SavePlot("First Test.png", useDefaultOutputFolder=True)
plt.show()
lendres.Plotting.SavePlot("Second Test.png", figure=figure, useDefaultOutputFolder=True)

#lendres.Plotting.DeleteOutputDirectory()