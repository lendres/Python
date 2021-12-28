# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""

# Use this to import from another directory.
import sys
sys.path.insert(1, "..\\lendres\\")

import lendres

data = lendres.Data.LoadAndInspectData("data.csv")

categories = ["age", "bmi", "children", "charges"]
lendres.UnivariateAnalysis.MakeBoxAndHistogramPlots(data, categories, save=True)

categories = ["sex", "children", "smoker", "region"]
lendres.UnivariateAnalysis.MakePercentageBarPlots(data, categories, save=True)
