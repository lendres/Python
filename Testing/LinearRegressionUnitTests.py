# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:53:03 2022

@author: Lance
"""


from lendres.LinearRegressionHelper import LinearRegressionHelper


import pandas as pd
from IPython.display import display

inputFile = "insurance.csv"

data = pd.read_csv(inputFile)
data.info()

data = pd.get_dummies(data, columns=["region", "sex", "smoker"], drop_first=True)

linearRegressionHelper = LinearRegressionHelper(data)
data.info()

linearRegressionHelper.SplitData("charges", 0.3)
linearRegressionHelper.CreateModel()

print("\n")
display(linearRegressionHelper.GetModelCoefficients())

print("\n")
display(linearRegressionHelper.GetModelPerformanceScores())