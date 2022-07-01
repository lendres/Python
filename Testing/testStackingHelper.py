"""
Created on December 27, 2021
@author: Lance A. Endres
"""
from sklearn.ensemble                     import StackingClassifier

import DataSetLoading

from lendres.ConsoleHelper                import ConsoleHelper
from lendres.AdaBoostHelper               import AdaBoostHelper
from lendres.GradientBoostingHelper       import GradientBoostingHelper
from lendres.XGradientBoostingHelper      import XGradientBoostingHelper
from lendres.StackingHelper               import StackingHelper

import unittest

class TestStackingHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCreditData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED, dropFirst=False)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper         = TestStackingHelper.dataHelper.Copy(deep=True)
        self.dataHelper.SplitData(TestStackingHelper.dependentVariable, 0.3, stratify=False)


    def testResults(self):
        estimator1       = AdaBoostHelper(self.dataHelper)
        estimator1.dataHelper.SplitData(TestStackingHelper.dependentVariable, 0.3, stratify=False)
        estimator1.Fit()

        estimator2       = GradientBoostingHelper(self.dataHelper)
        estimator2.dataHelper.SplitData(TestStackingHelper.dependentVariable, 0.3, stratify=False)
        estimator2.Fit()

        finalEstimator   = GradientBoostingHelper(self.dataHelper)
        finalEstimator.dataHelper.SplitData(TestStackingHelper.dependentVariable, 0.3, stratify=False)
        finalEstimator.Fit()

        estimators       = [("AdaBoost", estimator1.model), ("Gradient Boost", estimator2.model)]
        final_estimator  = finalEstimator.model

        self.regressionHelper   = StackingHelper(self.dataHelper, StackingClassifier(estimators=estimators, final_estimator=final_estimator))
        self.regressionHelper.FitPredict()

        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")
        result = self.regressionHelper.GetConfusionMatrix(dataSet="testing")
        self.assertAlmostEqual(result[1, 1], 28)

        self.regressionHelper.DisplayModelPerformanceScores(final=True)
        result = self.regressionHelper.GetModelPerformanceScores(final=True)
        self.assertAlmostEqual(result.loc["Testing", "Recall"], 0.3255, places=3)


if __name__ == "__main__":
    unittest.main()