# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:53:03 2022

@author: Lance
"""


from lendres.LogisticRegressionHelper import DecisionTreeHelper


import pandas as pd
from IPython.display import display

inputFile = ".csv"

data = pd.read_csv(inputFile)



regressionHelper = DecisionTreeHelper(data)

#columnAsNumeric = regressionHelper.ConvertCategoryToNumeric("Status", "Abnormal")
#data.info()

regressionHelper.SplitData(columnAsNumeric, 0.3)
regressionHelper.CreateModel()

#print("\n")
#display(regressionHelper.GetModelCoefficients())


regressionHelper.PlotConfusionMatrix(dataSet="training")
regressionHelper.PlotConfusionMatrix(dataSet="test")

print("\n")
display(regressionHelper.GetModelPerformanceScores())


regressionHelper.CreateRocCurvePlot()
regressionHelper.CreateRocCurvePlot("test")
regressionHelper.CreateRocCurvePlot("both")