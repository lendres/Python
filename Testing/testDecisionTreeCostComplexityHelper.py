"""
Created on December 27, 2021
@author: Lance
"""
#from   IPython.display                             import display

import DataSetLoading
from   lendres.DecisionTreeHelper                   import DecisionTreeHelper
from   lendres.DecisionTreeCostComplexityHelper     import DecisionTreeCostComplexityHelper

import unittest

class TestDecisionTreeCostComplexityHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.whichData = 0

        if cls.whichData == 0:
            cls.dataHelper, cls.dependentVariable = DataSetLoading.GetLoanModellingData()
        elif cls.whichData == 1:
            cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCreditData()


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper         = TestDecisionTreeCostComplexityHelper.dataHelper.Copy(deep=True)
        self.dataHelper.SplitData(TestDecisionTreeCostComplexityHelper.dependentVariable, 0.3, stratify=True)

        self.regressionHelper   = DecisionTreeCostComplexityHelper(self.dataHelper, DecisionTreeHelper.CreateDefaultModel())


    def testCostComplexityPruningModel(self):
        self.regressionHelper.Fit()
        self.regressionHelper.CreateCostComplexityPruningModel("recall")

        self.regressionHelper.Predict()
        result = self.regressionHelper.GetModelPerformanceScores(final=True)

        if TestDecisionTreeCostComplexityHelper.whichData == 0:
            self.assertAlmostEqual(result.loc["Training", "Accuracy"], 1.0000, places=3)
            self.assertAlmostEqual(result.loc["Testing", "Recall"], 0.8765, places=3)
        elif TestDecisionTreeCostComplexityHelper.whichData == 1:
            self.assertAlmostEqual(result.loc["Training", "Accuracy"], 0.7671, places=3)
            self.assertAlmostEqual(result.loc["Testing", "Recall"], 0.5333, places=3)


    def testCostComplexityPruningPlots(self):
        scoreMethod = "recall"
        #scoreMethod = "precision"
        self.regressionHelper.Fit()
        self.regressionHelper.CreateCostComplexityPruningModel(scoreMethod)

        self.regressionHelper.CreateImpunityVersusAlphaPlot()
        self.regressionHelper.CreateAlphasVersusScoresPlot(scoreMethod)

        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="training")
        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")
        result = self.regressionHelper.GetConfusionMatrix(dataSet="testing")
        if TestDecisionTreeCostComplexityHelper.whichData == 0:
            self.assertEqual(result[0, 1], 14)
            self.assertEqual(result[1, 1], 71)
        elif TestDecisionTreeCostComplexityHelper.whichData == 1:
            self.assertEqual(result[0, 1], 36)
            self.assertEqual(result[1, 1], 48)


if __name__ == "__main__":
    unittest.main()