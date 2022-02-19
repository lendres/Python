# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:53:03 2022

@author: Lance
"""
import pandas as pd
from IPython.display import display

from lendres.LogisticRegressionHelper import LogisticRegressionHelper

import unittest

class TestLogisticRegressionHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inputFile = "backpain.csv"
        #cls.data = lendres.Data.LoadAndInspectData(inputFile)
        cls.data   = pd.read_csv(inputFile)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.data             = TestLogisticRegressionHelper.data.copy(deep=True)
        self.regressionHelper = LogisticRegressionHelper(self.data)

        columnAsNumeric       = self.regressionHelper.ConvertCategoryToNumeric("Status", "Abnormal")
        self.regressionHelper.SplitData(columnAsNumeric, 0.3)
        self.regressionHelper.CreateModel()


    def testStandardPlots(self):

        self.regressionHelper.CreateRocCurvePlot()
        self.regressionHelper.CreateRocCurvePlot("testing")
        self.regressionHelper.CreateRocCurvePlot("both")

    def testPredictWithThreashold(self):
        self.regressionHelper.PredictWithThreashold(0.5)
        result = self.regressionHelper.GetModelPerformanceScores()
        self.assertAlmostEqual(result.loc["Training", "Accuracy"], 0.834101, places=6)
        self.assertAlmostEqual(result.loc["Testing", "Recall"], 0.878788, places=6)

        # Test a separate threashold to be sure we get different values.
        self.regressionHelper.PredictWithThreashold(0.8)
        result = self.regressionHelper.GetModelPerformanceScores()
        self.assertAlmostEqual(result.loc["Training", "Accuracy"], 0.783410, places=6)
        self.assertAlmostEqual(result.loc["Testing", "Recall"], 0.712121, places=6)


if __name__ == "__main__":
    unittest.main()