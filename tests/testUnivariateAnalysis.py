"""
Created on December 27, 2021
@author: Lance A. Endres
"""
import DataSetLoading

from   lendres.data.UnivariateAnalysis                               import UnivariateAnalysis
from   lendres.plotting.PlotMaker                                    import PlotMaker

import unittest


class TestUnivariateAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetInsuranceData(encode=False)


    def setUp(self):
        self.dataHelper = TestUnivariateAnalysis.dataHelper.Copy()


    def testBoxAndHistorgramPlot(self):
        categories = ["bmi", "charges"]
        PlotMaker.ApplyPlotToEachCategory(self.dataHelper.data, categories, UnivariateAnalysis.CreateBoxAndHistogramPlot)


    def testCreateCountPlot(self):
        categories = ["children", "smoker"]
        PlotMaker.ApplyPlotToEachCategory(self.dataHelper.data, categories, UnivariateAnalysis.CreateCountFigure)


    def testCreatBoxPlot(self):
        UnivariateAnalysis.CreateBoxPlot(self.dataHelper.data, "charges")


if __name__ == "__main__":
    unittest.main()