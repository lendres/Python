"""
Created on December 27, 2021
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
        PlotHelper.ApplyPlotToEachCategory(self.dataHelper.data, categories, UnivariateAnalysis.CreateBoxAndHistogramPlot)

        categories = ["children", "smoker"]
        PlotHelper.ApplyPlotToEachCategory(self.dataHelper.data, categories, UnivariateAnalysis.CreatePercentageBarPlot)

        UnivariateAnalysis.CreateBoxPlot(self.dataHelper.data, "charges")


if __name__ == "__main__":
    unittest.main()