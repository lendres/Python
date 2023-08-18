"""
Created on December 27, 2021
@author: Lance A. Endres
"""
import DataSetLoading

from   lendres.ConsoleHelper                     import ConsoleHelper
from   lendres.BivariateAnalysis                 import BivariateAnalysis
from   lendres.plotting.PlotMaker                import PlotMaker

import unittest


# Some of these tests take a long time to run.  Use this to skip some.  Useful for testing
# new unit tests so you don't have to run them all to see if the new one works.
skipTests = 0
if skipTests:
    skippedTests = ["Pair Plots", "Heat Maps"]
else:
    skippedTests = []


class TestBivariateAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.insuranceDataHelper, cls.insuranceDependentVariable = DataSetLoading.GetInsuranceData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED, encode=False)
        cls.cardioDataHelper,    cls.cardioDependentVariable    = DataSetLoading.GetCardioGoodFitnessData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED)


    def setUp(self):
        self.insuranceDataHelper = self.insuranceDataHelper.Copy()
        self.cardioDataHelper    = self.cardioDataHelper.Copy()


    @unittest.skipIf("Heat Maps" in skippedTests, "Skipped pair plots unit test.")
    def testHeatMapPlots(self):
        BivariateAnalysis.CreateBivariateHeatMap(self.insuranceDataHelper.data)

        columns = ["age", "charges"]
        BivariateAnalysis.CreateBivariateHeatMap(self.insuranceDataHelper.data, columns)


    @unittest.skipIf("Pair Plots" in skippedTests, "Skipped pair plots unit test.")
    def testPairPlots(self):
        BivariateAnalysis.CreateBivariatePairPlot(self.insuranceDataHelper.data)

        BivariateAnalysis.CreateBivariatePairPlot(self.insuranceDataHelper.data, hue="sex")

        columns = ["age", "bmi"]
        BivariateAnalysis.CreateBivariatePairPlot(self.insuranceDataHelper.data, columns)
        BivariateAnalysis.CreateBivariatePairPlot(self.insuranceDataHelper.data, columns, hue="sex")

        columns = list(self.insuranceDataHelper.data.columns)
        BivariateAnalysis.CreateBivariatePairPlot(self.insuranceDataHelper.data, columns, hue="sex")


    def testPlotComparisonByCategory(self):
        BivariateAnalysis.CreateScatterPlotComparisonByCategory(self.insuranceDataHelper.data, "age", "charges", "sex")


    def testCreateBarPlot(self):
        PlotMaker.CreateCountFigure(self.cardioDataHelper.data, "Product", xLabelRotation=45)
        PlotMaker.CreateCountFigure(self.cardioDataHelper.data, "Product", "Gender")


    def testProportionalData(self):
        BivariateAnalysis.CreateComparisonPercentageBarPlot(self.cardioDataHelper.data, "Product", ["TM498", "TM798"], "Gender")


    def testCreateStackedPercentageBarPlot(self):
        BivariateAnalysis.CreateStackedPercentageBarPlot(self.cardioDataHelper.data, "Product", "Gender")


    def testGetCrossTabulatedValueCounts(self):
        result = BivariateAnalysis.GetCrossTabulatedValueCounts(self.cardioDataHelper.data, "Product", "Gender")
        self.cardioDataHelper.consoleHelper.Display(result, verboseLevel=ConsoleHelper.VERBOSEALL)
        self.assertEqual(result.loc["TM195", "Female"], 40)


    def testPlottingByTarget(self):
        # Test where sort value is text.
        BivariateAnalysis.CreateDistributionByTargetPlot(self.insuranceDataHelper.data, "bmi", "sex")
        BivariateAnalysis.CreateBoxPlotByTarget(self.insuranceDataHelper.data, "bmi", "sex")

        # Test where sort value is numerical.  Also tests where sort value has more than two categories.
        BivariateAnalysis.CreateDistributionByTargetPlot(self.insuranceDataHelper.data, "bmi", "children")
        BivariateAnalysis.CreateBoxPlotByTarget(self.insuranceDataHelper.data, "bmi", "children")


if __name__ == "__main__":
    unittest.main()