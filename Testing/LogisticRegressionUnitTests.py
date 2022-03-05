# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:53:03 2022

@author: Lance
"""
#from IPython.display import display

from lendres.ConsoleHelper import ConsoleHelper
from lendres.DataHelper import DataHelper
from lendres.LogisticRegressionHelper import LogisticRegressionHelper

import unittest

class TestLogisticRegressionHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inputFile       = "backpain.csv"

        consoleHelper   = ConsoleHelper(verboseLevel=ConsoleHelper.VERBOSENONE)
        cls.dataHelper  = DataHelper(consoleHelper=consoleHelper)
        cls.dataHelper.LoadAndInspectData(inputFile)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper       = TestLogisticRegressionHelper.dataHelper.Copy(deep=True)
        self.regressionHelper = LogisticRegressionHelper(self.dataHelper)

        columnAsNumeric       = self.regressionHelper.ConvertCategoryToNumeric("Status", "Abnormal")
        self.regressionHelper.SplitData(columnAsNumeric, 0.3)
        self.regressionHelper.CreateModel()


    def testStandardPlots(self):
        self.regressionHelper.CreateRocCurvePlot()
        self.regressionHelper.CreateRocCurvePlot("testing")
        self.regressionHelper.CreateRocCurvePlot("both")


    def testPredictWithThreshold(self):
        self.regressionHelper.PredictWithThreshold(0.5)
        result = self.regressionHelper.GetModelPerformanceScores()
        self.assertAlmostEqual(result.loc["Training", "Accuracy"], 0.834101, places=6)
        self.assertAlmostEqual(result.loc["Testing", "Recall"], 0.878788, places=6)

        # Test a separate threashold to be sure we get different values.
        self.regressionHelper.PredictWithThreshold(0.8)
        result = self.regressionHelper.GetModelPerformanceScores()
        #display(result)
        self.assertAlmostEqual(result.loc["Training", "Accuracy"], 0.783410, places=6)
        self.assertAlmostEqual(result.loc["Testing", "Recall"], 0.712121, places=6)


    def testGetOdds(self):
        result = self.regressionHelper.GetOdds(sort=True)
        #display(result)
        self.assertAlmostEqual(result.loc["pelvic_incidence", "Odds"], 1.035757, places=6)

if __name__ == "__main__":
    unittest.main()