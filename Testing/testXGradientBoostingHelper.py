"""
Created on December 27, 2021
@author: Lance A. Endres
"""
from   IPython.display                                          import display

import DataSetLoading
from   lendres.ConsoleHelper                                    import ConsoleHelper
from   lendres.XGradientBoostingHelper                          import XGradientBoostingHelper

import unittest

class TestXGradientBoostingHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCreditData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED, dropFirst=False)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper         = TestXGradientBoostingHelper.dataHelper.Copy(deep=True)
        self.dataHelper.SplitData(TestXGradientBoostingHelper.dependentVariable, 0.3, stratify=True)

        self.regressionHelper   = XGradientBoostingHelper(self.dataHelper)


    # def testResults(self):
    #     self.regressionHelper.FitPredict()
    #     self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")
    #     result = self.regressionHelper.GetConfusionMatrix(dataSet="testing")
    #     self.assertAlmostEqual(result[1, 1], 48)
    #     result = self.regressionHelper.GetModelPerformanceScores(final=True)
    #     #display(result)
    #     self.assertAlmostEqual(result.loc["Testing", "Recall"], 0.5333, places=3)


if __name__ == "__main__":
    unittest.main()