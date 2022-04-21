# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
import DataSetLoading

from lendres.ConsoleHelper           import ConsoleHelper
from lendres.AdaBoostHelper          import AdaBoostHelper
from lendres.GradientBoostingHelper  import GradientBoostingHelper
from lendres.XGradientBoostingHelper import XGradientBoostingHelper
from lendres.StackingHelper          import StackingHelper

import unittest

class TestStackingHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCreditData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED, dropFirst=False)
        #cls.dataHelper.PrintFinalDataSummary()

    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper         = TestStackingHelper.dataHelper.Copy(deep=True)
        self.regressionHelper   = StackingHelper(self.dataHelper)

        self.regressionHelper.dataHelper.SplitData(TestStackingHelper.dependentVariable, 0.3, stratify=False)


    def testResults(self):
        estimator1       = AdaBoostHelper(self.dataHelper)
        estimator1.dataHelper.SplitData(TestStackingHelper.dependentVariable, 0.3, stratify=False)
        estimator1.CreateModel()

        estimator2       = GradientBoostingHelper(self.dataHelper)
        estimator2.dataHelper.SplitData(TestStackingHelper.dependentVariable, 0.3, stratify=False)
        estimator2.CreateModel()

        finalEstimator   = XGradientBoostingHelper(self.dataHelper)
        finalEstimator.dataHelper.SplitData(TestStackingHelper.dependentVariable, 0.3, stratify=False)
        finalEstimator.CreateModel()

        estimators       = [('AdaBoost', estimator1.model), ('Gradient Boost', estimator2.model)]
        final_estimator  = finalEstimator.model

        self.regressionHelper.CreateModel(estimators=estimators, final_estimator=final_estimator)
        self.regressionHelper.Predict()

        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")
        result = self.regressionHelper.GetConfusionMatrix(dataSet="testing")
        self.assertAlmostEqual(result[1, 1], 28)

        self.regressionHelper.DisplayModelPerformanceScores(final=True)
        result = self.regressionHelper.GetModelPerformanceScores(final=True)
        self.assertAlmostEqual(result.loc["Testing", "Recall"], 0.3255, places=3)


if __name__ == "__main__":
    unittest.main()