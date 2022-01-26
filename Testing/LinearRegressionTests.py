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

linearRegressionHelper = LinearRegressionHelper()

linearRegressionHelper.CreateLinearModel(data, "charges", 0.3)