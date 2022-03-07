# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
import DataSetLoading
import lendres

import unittest

class TestBivariateAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetInsuranceData(encode=False)


    def setUp(self):
        self.dataHelper = TestBivariateAnalysis.dataHelper.Copy(deep=True)


    def testPlots(self):
        lendres.BivariateAnalysis.CreateBiVariateHeatMap(self.dataHelper.data)
        lendres.BivariateAnalysis.CreateBiVariatePairPlot(self.dataHelper.data)

        columns = ["age", "charges"]
        lendres.BivariateAnalysis.CreateBiVariateHeatMap(self.dataHelper.data, columns)
        lendres.BivariateAnalysis.CreateBiVariatePairPlot(self.dataHelper.data, columns)


if __name__ == "__main__":
    unittest.main()