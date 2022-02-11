# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:53:03 2022

@author: Lance
"""


from lendres.LogisticRegressionHelper import LogisticRegressionHelper


import pandas as pd
from IPython.display import display

inputFile = "backpain.csv"

data = pd.read_csv(inputFile)



regressionHelper = LogisticRegressionHelper(data)

columnAsNumeric = regressionHelper.ConvertCategoryToNumeric("Status", "Abnormal")
data.info()

regressionHelper.SplitData(columnAsNumeric, 0.3)
regressionHelper.CreateModel()

print("\n")
display(regressionHelper.GetModelCoefficients())

# print("\n")
#display(regressionHelper.GetModelPerformanceScores())

regressionHelper.PlotConfusionMatrix(dataSet="training")
regressionHelper.PlotConfusionMatrix(dataSet="test")