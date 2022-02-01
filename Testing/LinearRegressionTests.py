# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:53:03 2022

@author: Lance
"""

import lendres

from lendres.LinearRegressionHelper import LinearRegressionHelper


import pandas as pd
from IPython.display import display

inputFile = "insurance.csv"

data = pd.read_csv(inputFile)
data.info()

data = pd.get_dummies(data, columns=["region", "sex", "smoker"], drop_first=True)

linearRegressionHelper = LinearRegressionHelper()
data.info()

linearRegressionHelper.CreateLinearModel(data, "charges", 0.3)
linearRegressionHelper.DisplayModelCoefficients()