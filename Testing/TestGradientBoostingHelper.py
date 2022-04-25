"""
Created on December 27, 2021
@author: Lance
"""
import DataSetLoading
from lendres.ConsoleHelper             import ConsoleHelper
from lendres.GradientBoostingHelper    import GradientBoostingHelper

import unittest

class TestGradientBoostingHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCreditData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED, dropFirst=False)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper         = TestGradientBoostingHelper.dataHelper.Copy(deep=True)
        self.dataHelper.SplitData(TestGradientBoostingHelper.dependentVariable, 0.3, stratify=True)

        self.regressionHelper   = GradientBoostingHelper(self.dataHelper)


    def testResults(self):
        self.regressionHelper.FitPredict()

        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")
        result = self.regressionHelper.GetConfusionMatrix(dataSet="testing")
        self.assertAlmostEqual(result[1, 1], 40)

        self.regressionHelper.DisplayModelPerformanceScores(final=True)
        result = self.regressionHelper.GetModelPerformanceScores(final=True)
        self.assertAlmostEqual(result.loc["Testing", "Recall"], 0.4444, places=3)


if __name__ == "__main__":
    unittest.main()