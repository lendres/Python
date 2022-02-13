# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:53:03 2022

@author: Lance
"""


from lendres.LogisticRegressionHelper import LogisticRegressionHelper


import pandas as pd
from IPython.display import display

inputFile = "backpain.csv"

data             = pd.read_csv(inputFile)
regressionHelper = LogisticRegressionHelper(data)

# Fix a column.
columnAsNumeric = regressionHelper.ConvertCategoryToNumeric("Status", "Abnormal")
data.info()

# Split the data and create the model.
regressionHelper.SplitData(columnAsNumeric, 0.3)
regressionHelper.CreateModel()

print("\n")
display(regressionHelper.GetModelCoefficients())


# Plot confusion matrices.
print("\nTraining data confusion matrix:")
display(regressionHelper.GetConfusionMatrix(dataSet="training"))
print("\nTraining data confusion matrix:")
display(regressionHelper.GetConfusionMatrix(dataSet="testing"))

regressionHelper.CreateConfusionMatrixPlot(dataSet="training")
regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")

print("\n")
regressionHelper.PredictWithThreashold(0.5)
display(regressionHelper.GetModelPerformanceScores())

#           Accuracy    Recall  Precision        F1
# Training  0.834101  0.868056   0.880282  0.874126
# Test      0.870968  0.878788   0.935484  0.906250

regressionHelper.CreateRocCurvePlot()
regressionHelper.CreateRocCurvePlot("testing")
regressionHelper.CreateRocCurvePlot("both")