# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
import DataSetLoading

from lendres.ConsoleHelper           import ConsoleHelper
from lendres.BivariateAnalysis       import BivariateAnalysis

import unittest

class TestBivariateAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetInsuranceData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED, encode=False)


    def setUp(self):
        self.dataHelper = TestBivariateAnalysis.dataHelper.Copy(deep=True)


    def testHeatMapPlots(self):
        BivariateAnalysis.CreateBivariateHeatMap(self.dataHelper.data)

        columns = ["age", "charges"]
        BivariateAnalysis.CreateBivariateHeatMap(self.dataHelper.data, columns)


    def testPairPlots(self):
        BivariateAnalysis.CreateBivariatePairPlot(self.dataHelper.data)

        columns = ["age", "charges"]
        BivariateAnalysis.CreateBivariatePairPlot(self.dataHelper.data, columns)


    def testPlotComparisonByCategory(self):
        BivariateAnalysis.PlotComparisonByCategory(self.dataHelper.data, "age", "charges", "sex", "Sorted by Sex")


if __name__ == "__main__":
    unittest.main()