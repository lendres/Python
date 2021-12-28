# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""

# Use this to import from another directory.
import sys
sys.path.insert(1, "..\\lendres\\")

import lendres.Data
import lendres.Console
import lendres.UnivariableAnalysis

data = lendres.Data.LoadAndInspectData("data.csv")

categories = ["age", "bmi", "children", "charges"]
lendres.UnivariableAnalysis.MakeBoxAndHistogramPlots(data, categories)

categories = ["sex", "children", "smoker", "region"]
lendres.UnivariableAnalysis.MakePercentageBarPlots(data, categories)
