# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
import DataSetLoading
import lendres
from lendres.PlotHelper import PlotHelper

import unittest

class TestUnivariateAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetInsuranceData(encode=False)


    def setUp(self):
        self.dataHelper = TestUnivariateAnalysis.dataHelper.Copy(deep=True)


    def testPlots(self):
        saveImages = False

        categories = ["bmi", "charges"]
        PlotHelper.ApplyPlotToEachCategory(lendres.UnivariateAnalysis.CreateBoxAndHistogramPlot, self.dataHelper.data, categories, save=saveImages)

        categories = ["children", "smoker"]
        PlotHelper.ApplyPlotToEachCategory(lendres.UnivariateAnalysis.CreatePercentageBarPlot, self.dataHelper.data, categories, save=saveImages)

        lendres.UnivariateAnalysis.CreateBoxPlot(self.dataHelper.data, "charges")


if __name__ == "__main__":
    unittest.main()