"""
Created on January 26, 2022
@author: Lance A. Endres
"""
from   sklearn.linear_model                                          import LogisticRegression

import DataSetLoading
from   lendres.io.ConsoleHelper                                      import ConsoleHelper
from   lendres.modeling.CategoricalRegressionHelper                  import CategoricalRegressionHelper

import unittest

class TestCategoricalRegressionHelper(unittest.TestCase):
    #verboseLevel = ConsoleHelper.VERBOSENONE
    verboseLevel = ConsoleHelper.VERBOSETESTING
    #verboseLevel = ConsoleHelper.VERBOSEREQUESTED
    #verboseLevel = ConsoleHelper.VERBOSEIMPORTANT

    @classmethod
    def setUpClass(cls):
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetBackPainData(verboseLevel=cls.verboseLevel)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper       = TestCategoricalRegressionHelper.dataHelper.Copy()
        self.dataHelper.SplitData(TestCategoricalRegressionHelper.dependentVariable, 0.3, stratify=True)

        self.regressionHelper = CategoricalRegressionHelper(self.dataHelper, LogisticRegression(solver="liblinear", random_state=1))


        # Fake a model so we have output to use.
        self.regressionHelper.Fit()


    def testConfusionMatrices(self):
        self.regressionHelper.Predict()
        result = self.regressionHelper.GetConfusionMatrix(dataSet="training")
        self.assertEqual(result.tolist(), [[53,  17], [18, 129]])

        result = self.regressionHelper.GetConfusionMatrix(dataSet="testing")
        self.assertEqual(result.tolist(), [[25,  5], [9, 54]])


    def testStandardPlots(self):
        self.regressionHelper.Predict()
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="training")
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")


    def testModelCoefficients(self):
        self.regressionHelper.Predict()
        result = self.regressionHelper.GetModelCoefficients()
        self.assertAlmostEqual(result.loc["pelvic_incidence", "Coefficients"], 0.02318, places=3)
        self.assertAlmostEqual(result.loc["Intercept", "Coefficients"], 1.1290, places=3)


    def testPredictionsNotCalculated(self):
        # Cannot call "FitPredict" is test setup because we want to test this exception.
        self.assertRaises(Exception, self.regressionHelper.GetModelPerformanceScores)


    def testModelPerformanceScores(self):
        self.regressionHelper.Predict()
        result = self.regressionHelper.GetModelPerformanceScores(final=True)
        self.assertAlmostEqual(result.loc["Training", "Accuracy"], 0.8387, places=3)
        self.assertAlmostEqual(result.loc["Testing", "Recall"], 0.8571, places=3)


    def testSplitComparisons(self):
        self.regressionHelper.Predict()
        result = self.regressionHelper.dataHelper.GetSplitComparisons()
        self.regressionHelper.dataHelper.consoleHelper.Display(result, ConsoleHelper.VERBOSEREQUESTED)
        self.assertEqual(result.loc[1, "Testing"], "63 (67.74%)")


if __name__ == "__main__":
    unittest.main()