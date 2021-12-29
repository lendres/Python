# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
import lendres

saveImages = False

data = lendres.Data.LoadAndInspectData("data.csv")

categories = ["age", "bmi", "children", "charges"]
lendres.UnivariateAnalysis.MakeBoxAndHistogramPlots(data, categories, save=saveImages)

categories = ["sex", "children", "smoker", "region"]
lendres.UnivariateAnalysis.MakePercentageBarPlots(data, categories, save=saveImages)
