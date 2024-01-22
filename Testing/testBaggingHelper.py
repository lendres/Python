"""
Created on December 27, 2021
@author: Lance A. Endres
"""
from   sklearn.linear_model                                          import LogisticRegression

import DataSetLoading
from   lendres.io.ConsoleHelper                                      import ConsoleHelper
from   lendres.BaggingHelper                                         import BaggingHelper

import unittest

class TestBaggingHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        verboseLevel = ConsoleHelper.VERBOSEREQUESTED
        verboseLevel = ConsoleHelper.VERBOSETESTING
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCreditData(verboseLevel=verboseLevel, dropFirst=False)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper         = TestBaggingHelper.dataHelper.Copy()
        self.dataHelper.SplitData(TestBaggingHelper.dependentVariable, 0.3, stratify=True)

        self.regressionHelper   = BaggingHelper(self.dataHelper)


    def testDefaultResults(self):
        self.regressionHelper.FitPredict()
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")

        result = self.regressionHelper.GetConfusionMatrix(dataSet="testing")
        self.assertAlmostEqual(result[1, 1], 43)

        result = self.regressionHelper.GetModelPerformanceScores()
        self.assertAlmostEqual(result.loc["Training", "Recall"], 0.9428, places=3)


    def testLogisticRegressionClassifier(self):
        baseEstimator = LogisticRegression(solver='liblinear', max_iter=1000, random_state=1)
        self.regressionHelper   = BaggingHelper(self.dataHelper, BaggingHelper.CreateDefaultModel(estimator=baseEstimator))

        self.regressionHelper.FitPredict()
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")

        result = self.regressionHelper.GetConfusionMatrix(dataSet="testing")
        self.assertAlmostEqual(result[1, 1], 32)

        result = self.regressionHelper.GetModelPerformanceScores()
        self.assertAlmostEqual(result.loc["Training", "Recall"], 0.33, places=1)
        self.regressionHelper.DisplayModelPerformanceScores()


if __name__ == "__main__":
    unittest.main()