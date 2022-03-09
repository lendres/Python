# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
import DataSetLoading

from lendres.PlotHelper import PlotHelper
from lendres.UnivariateAnalysis import UnivariateAnalysis

import unittest

class TestUnivariateAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetInsuranceData(encode=False)


    def setUp(self):
        self.dataHelper = TestUnivariateAnalysis.dataHelper.Copy(deep=True)


    def testPlots(self):
        categories = ["bmi", "charges"]
        PlotHelper.ApplyPlotToEachCategory(UnivariateAnalysis.CreateBoxAndHistogramPlot, self.dataHelper.data, categories)

        categories = ["children", "smoker"]
        PlotHelper.ApplyPlotToEachCategory(UnivariateAnalysis.CreatePercentageBarPlot, self.dataHelper.data, categories)

        UnivariateAnalysis.CreateBoxPlot(self.dataHelper.data, "charges")


if __name__ == "__main__":
    unittest.main()