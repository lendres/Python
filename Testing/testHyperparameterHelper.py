"""
Created on December 27, 2021
@author: Lance A. Endres
"""
import numpy                                     as np
from   IPython.display                           import display
from   sklearn                                   import metrics

from   sklearn.tree                              import DecisionTreeClassifier
from   sklearn.ensemble                          import AdaBoostClassifier

import DataSetLoading
from   lendres.ConsoleHelper                     import ConsoleHelper
from   lendres.ModelHelper                       import ModelHelper
from   lendres.DecisionTreeHelper                import DecisionTreeHelper
from   lendres.BaggingHelper                     import BaggingHelper
from   lendres.RandomForestHelper                import RandomForestHelper
from   lendres.AdaBoostHelper                    import AdaBoostHelper
from   lendres.GradientBoostingHelper            import GradientBoostingHelper
from   lendres.XGradientBoostingHelper           import XGradientBoostingHelper
from   lendres.HyperparameterHelper              import HyperparameterHelper

import unittest

# Some of these tests take a long time to run.  Use this to skip some.  Useful for testing
# new unit tests so you don't have to run them all to see if the new one works.
skipTests = 1
if skipTests:
    #skippedTests = ["Decision Tree", "Bagging", "Random Forest", "AdaBoost", "Gradient Boosting", "X Gradient Boosting"]
    #skippedTests = ["AdaBoost", "X Gradient Boosting"]
    skippedTests = ["X Gradient Boosting"]
else:
    skippedTests = []


class TestHyperparameterHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.regresionHelpers = []

        #VERBOSETESTING
        #VERBOSEREQUESTED
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCreditData(verboseLevel=ConsoleHelper.VERBOSETESTING, dropFirst=False)

        if skipTests:
            print("\nThe following tests have been skipped:")
            for test in skippedTests:
                print("    ", test)
            print("\n")


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper             = TestHyperparameterHelper.dataHelper.Copy(deep=True)


    @unittest.skipIf("Decision Tree" in skippedTests, "Skipped decision tree unit test.")
    def testDecisionTreeClassifier(self):
        parameters = {
            "max_depth"             : np.arange(1, 3),
            "min_samples_leaf"      : [2, 5, 7],
            "max_leaf_nodes"        : [2, 5, 10],
            "criterion"             : ["entropy", "gini"]
        }

        self.dataHelper.SplitData(TestHyperparameterHelper.dependentVariable, 0.2, 0.25, stratify=True)
        self.regressionHelper             = DecisionTreeHelper(self.dataHelper)
        self.regressionHelper.description = "Decision Tree"
        scores, confusionMatrix = self.RunClassifier(parameters, True)

        self.assertAlmostEqual(confusionMatrix[1, 1], 34)


    @unittest.skipIf("Bagging" in skippedTests, "Skipped bagging unit test.")
    def testBaggingClassifier(self):
        parameters = {
            "max_samples"  : [0.7, 0.8],
            "max_features" : [0.7, 0.9],
            "n_estimators" : [10,  20]
        }

        self.dataHelper.SplitData(TestHyperparameterHelper.dependentVariable, 0.2, 0.25, stratify=True)
        self.regressionHelper             = BaggingHelper(self.dataHelper)
        self.regressionHelper.description = "Bagging"
        scores, confusionMatrix           = self.RunClassifier(parameters, True)
        self.assertAlmostEqual(confusionMatrix[1, 1], 27)


    @unittest.skipIf("Random Forest" in skippedTests, "Skipped random forest unit test.")
    def testRandomForestClassifier(self):
        parameters = {
            "n_estimators"     : [10, 20],
            "min_samples_leaf" : [5, 8],
            "max_features"     : [0.2, 0.7],
            "max_samples"      : [0.3, 0.7]
        }

        self.dataHelper.SplitData(TestHyperparameterHelper.dependentVariable, 0.3, stratify=True)
        self.regressionHelper             = RandomForestHelper(self.dataHelper)
        self.regressionHelper.description = "Random Forest"
        scores, confusionMatrix           = self.RunClassifier(parameters, False)
        self.assertAlmostEqual(confusionMatrix[1, 1], 39)


    @unittest.skipIf("AdaBoost" in skippedTests, "Skipped adaboost unit test.")
    def testAdaBoostClassifier(self):
        parameters = {
            "base_estimator" : [DecisionTreeClassifier(max_depth=1, random_state=1),
                                DecisionTreeClassifier(max_depth=2, random_state=1)],
            "n_estimators"   : [10, 25],
            "learning_rate"  : [0.1, 0.5]
        }

        self.dataHelper.SplitData(TestHyperparameterHelper.dependentVariable, 0.2, 0.25, stratify=True)
        self.regressionHelper             = AdaBoostHelper(self.dataHelper)
        self.regressionHelper.description = "Adaboost"
        scores, confusionMatrix = self.RunClassifier(parameters, True)
        self.assertAlmostEqual(confusionMatrix[1, 1], 36)


    @unittest.skipIf("Gradient Boosting" in skippedTests, "Skipped gradient boosting unit test.")
    def testGradientBoostingClassifier(self):
        parameters = {
            "n_estimators" : [50, 100],
            "subsample"    : [0.8, 0.9],
            "max_features" : [0.7, 0.8]
        }

        self.dataHelper.SplitData(TestHyperparameterHelper.dependentVariable, 0.3, stratify=True)
        self.regressionHelper             = GradientBoostingHelper(self.dataHelper)
        self.regressionHelper.description = "Gradient Boosting"
        scores, confusionMatrix           = self.RunClassifier(parameters, False)
        self.assertAlmostEqual(confusionMatrix[1, 1], 43)


    @unittest.skipIf("X Gradient Boosting" in skippedTests, "Skipped extreme gradient boosting unit test.")
    def testXGradientBoostingClassifier(self):
        parameters = {
            "n_estimators"      : np.arange(10, 20, 5),
            "scale_pos_weight"  : [5],
            "subsample"         : [0.5, 0.9],
            "learning_rate"     : [0.2, 0.05],
            "gamma"             : [0, 3],
            "colsample_bytree"  : [0.5],
            "colsample_bylevel" : [0.5]
        }

        self.dataHelper.SplitData(TestHyperparameterHelper.dependentVariable, 0.2, 0.25, stratify=True)
        self.regressionHelper             = XGradientBoostingHelper(self.dataHelper)
        self.regressionHelper.description = "X Gradient Boost"
        scores, confusionMatrix           = self.RunClassifier(parameters, True)
        self.assertAlmostEqual(confusionMatrix[1, 1], 55)


    def testZComparison(self):
        # Needs to run after at least two of the other tests have been run.
        # The "Z" used after "test" and before the test name is used to make sure this is called at the end.  The
        # functions are run alphabetically.
        self.dataHelper.consoleHelper.PrintNewLine()
        self.dataHelper.consoleHelper.PrintSectionTitle("Model Comparisons")

        # Test getting a single score.
        ModelHelper.DisplayModelComparisons("Recall", TestHyperparameterHelper.regresionHelpers)

        # Test getting multiple scores and test using the print function.
        print("\n\n")
        ModelHelper.DisplayModelComparisons(["Accuracy", "Recall"], TestHyperparameterHelper.regresionHelpers)

        # Test using the internally saved model helpers of ModelHelpr.
        print("\n\n")
        ModelHelper.DisplayModelComparisons(["Accuracy", "Recall"])


    def testZPlot(self):
        # Needs to run after at least two of the other tests have been run.
        # The "Z" used after "test" and before the test name is used to make sure this is called at the end.  The
        # functions are run alphabetically.
        ModelHelper.CreateScorePlotForAllModels("F1", width=8)


    def testZZAdditionalOutput(self):
        parameters = {"base_estimator" : [DecisionTreeClassifier(max_depth=1, random_state=1),
                                          DecisionTreeClassifier(max_depth=2, random_state=1)],
                      "n_estimators"   : [10, 25],
                      "learning_rate"  : [0.1, 0.5]}

        self.dataHelper.SplitData(TestHyperparameterHelper.dependentVariable, 0.2, 0.25, stratify=True)
        self.regressionHelper             = AdaBoostHelper(self.dataHelper)
        self.regressionHelper.description = "Adaboost"

        self.hyperparameterHelper   = HyperparameterHelper(self.regressionHelper, metrics.recall_score, "grid", True)
        self.hyperparameterHelper.RunHypertuning(parameters, saveModel=False)
        self.hyperparameterHelper.RunHypertuning(parameters, saveModel=False)


    def RunClassifier(self, parameters, saveToModelHelper, testingOutput=False):
        self.dataHelper.consoleHelper.PrintNewLine()
        self.regressionHelper.PrintClassName()

        self.hyperparameterHelper   = HyperparameterHelper(self.regressionHelper, metrics.recall_score, "grid", testingOutput)
        self.hyperparameterHelper.RunHypertuning(parameters, saveModel=saveToModelHelper)

        scores = self.regressionHelper.GetModelPerformanceScores(final=True)

        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing", titlePrefix=self.regressionHelper.description)
        confusionMatrix = self.regressionHelper.GetConfusionMatrix(dataSet="testing")

        if saveToModelHelper:
            ModelHelper.SaveModelHelper(self.regressionHelper)
        else:
            TestHyperparameterHelper.regresionHelpers.append(self.regressionHelper)

        return scores, confusionMatrix


if __name__ == "__main__":
    unittest.main()