# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 07:49:25 2022

@author: Lance
"""

from IPython.display import display
import pandas as pd
import numpy as np

#import lendres

def DisplayModelCoefficients(regressionModel, xTrainingData):
    dataFrame = pd.DataFrame(
        np.append(regressionModel.coef_, regressionModel.intercept_),
        index = xTrainingData.columns.tolist() + ["Intercept"],
        columns = ["Coefficients"],
    )

    display(dataFrame)