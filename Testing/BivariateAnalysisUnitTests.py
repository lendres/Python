# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""

# Use this to import from another directory.

import lendres

data = lendres.Data.LoadAndInspectData("data.csv")
data['children'] = data['children'].astype('category')

lendres.BivariateAnalysis.CreateBiVariateHeatMap(data)
lendres.BivariateAnalysis.CreateBiVariatePairPlot(data)


columns = ["age", "charges"]
lendres.BivariateAnalysis.CreateBiVariateHeatMap(data, columns)
lendres.BivariateAnalysis.CreateBiVariatePairPlot(data, columns)