# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:53:03 2022

@author: Lance
"""
#from IPython.display import display
from sklearn.linear_model import LogisticRegression

import os

from lendres.ConsoleHelper import ConsoleHelper
from lendres.DataHelper import DataHelper
from lendres.CategoricalRegressionHelper import CategoricalRegressionHelper

import unittest

class TestCategoricalRegressionHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inputFile = "backpain.csv"

        inputFile       = os.path.join("Data", inputFile)

        consoleHelper   = ConsoleHelper(verboseLevel=ConsoleHelper.VERBOSENONE)
        cls.dataHelper  = DataHelper(consoleHelper=consoleHelper)
        cls.dataHelper.LoadAndInspectData(inputFile)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper       = TestCategoricalRegressionHelper.dataHelper.Copy(deep=True)
        self.regressionHelper = CategoricalRegressionHelper(self.dataHelper)

        columnAsNumeric       = self.regressionHelper.ConvertCategoryToNumeric("Status", "Abnormal")
        self.regressionHelper.SplitData(columnAsNumeric, 0.3)

        # Fake a model so we have output to use.
        self.regressionHelper.model = LogisticRegression(solver="liblinear", random_state=1)
        self.regressionHelper.model.fit(self.regressionHelper.xTrainingData, self.regressionHelper.yTrainingData.values.ravel())


    def testConfusionMatrices(self):
        result = self.regressionHelper.GetConfusionMatrix(dataSet="training")
        self.assertEqual(result.tolist(), [[53,  17], [18, 129]])

        result = self.regressionHelper.GetConfusionMatrix(dataSet="testing")
        self.assertEqual(result.tolist(), [[25,  5], [9, 54]])


    def testStandardPlots(self):
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="training")
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")


    def testModelCoefficients(self):
        result = self.regressionHelper.GetModelCoefficients()
        self.assertAlmostEqual(result.loc["pelvic_incidence", "Coefficients"], 0.02318, places=3)
        self.assertAlmostEqual(result.loc["Intercept", "Coefficients"], 1.1290, places=3)


    def testPredictionsNotCalculated(self):
        self.assertRaises(Exception, self.regressionHelper.GetModelPerformanceScores)


    def testModelPerformanceScores(self):
        self.regressionHelper.Predict()
        result = self.regressionHelper.GetModelPerformanceScores()
        self.assertAlmostEqual(result.loc["Training", "Accuracy"], 0.8387, places=3)
        self.assertAlmostEqual(result.loc["Testing", "Recall"], 0.8571, places=3)


    def testSplitComparisons(self):
        result = self.regressionHelper.GetSplitComparisons()
        self.assertEqual(result.loc["Testing", "Positive"], "63 (67.74%)")



if __name__ == "__main__":
    unittest.main()