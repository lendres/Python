# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
import numpy as np
#from IPython.display import display
from sklearn import metrics

import DataSetLoading
from lendres.ConsoleHelper import ConsoleHelper
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from lendres.DecisionTreeHelper import DecisionTreeHelper
from lendres.BaggingHelper import BaggingHelper
from lendres.RandomForestHelper import RandomForestHelper
from lendres.AdaBoostHelper import AdaBoostHelper
from lendres.GradientBoostingHelper import GradientBoostingHelper
from lendres.XGradientBoostingHelper import XGradientBoostingHelper
from lendres.HyperparameterHelper import HyperparameterHelper

import unittest

# Some of these tests take a long time to run.  Use this to skip some.  Useful for testing
# new unit tests so you don't have to run them all to see if the new one works.
skipTests = 1
if skipTests:
    #skippedTests = ["Decision Tree", "Bagging", "Random Forest", "AdaBoost", "Gradient Boosting", "X Gradient Boosting"]
    skippedTests = ["Decision Tree", "Bagging", "Random Forest", "AdaBoost", "Gradient Boosting"]
else:
    skippedTests = []

class TestHyperparameterHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCreditData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED,
                                                                             dropFirst=False)

        if skipTests:
            print("\nThe following tests have been skipped:")
            for test in skippedTests:
                print("    ", test)
            print("\n")

        #print("\nData size after cleaning:")
        #display(cls.dataHelper.data.shape)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper             = TestHyperparameterHelper.dataHelper.Copy(deep=True)


    @unittest.skipIf("Decision Tree" in skippedTests, "Skipped decision tree unit test.")
    def testDecisiionTreeClassifier(self):
        parameters = {"max_depth"             : np.arange(1, 3),
                      "min_samples_leaf"      : [2, 5, 7],
                      "max_leaf_nodes"        : [2, 5, 10],
                      "criterion"             : ["entropy", "gini"]}

        self.regressionHelper   = DecisionTreeHelper(self.dataHelper)
        scores, confusionMatrix = self.RunClassifier(parameters)
        self.assertAlmostEqual(confusionMatrix[1, 1], 48)


    @unittest.skipIf("Bagging" in skippedTests, "Skipped bagging unit test.")
    def testBaggingClassifier(self):
        parameters = {"max_samples"  : [0.7, 0.8],
                      "max_features" : [0.7, 0.9],
                      "n_estimators" : [10,  20]}

        self.regressionHelper   = BaggingHelper(self.dataHelper)
        scores, confusionMatrix = self.RunClassifier(parameters)
        self.assertAlmostEqual(confusionMatrix[1, 1], 48)


    @unittest.skipIf("Random Forest" in skippedTests, "Skipped random forest unit test.")
    def testRandomForestClassifier(self):
        parameters = {"n_estimators"     : [150, 200],
                      "min_samples_leaf" : np.arange(5, 8),
                      "max_features"     : np.arange(0.2, 0.7, 0.1),
                      "max_samples"      : np.arange(0.3, 0.7, 0.1)}

        self.regressionHelper   = RandomForestHelper(self.dataHelper)
        scores, confusionMatrix = self.RunClassifier(parameters)
        self.assertAlmostEqual(confusionMatrix[1, 1], 48)


    @unittest.skipIf("AdaBoost" in skippedTests, "Skipped adaboost unit test.")
    def testAdaBoostClassifier(self):
        parameters = {"base_estimator" : [DecisionTreeClassifier(max_depth=1, random_state=1),
                                          DecisionTreeClassifier(max_depth=2, random_state=1)],
                      "n_estimators"   : np.arange(10, 25),
                      "learning_rate"  : np.arange(0.1, 0.5)}

        self.regressionHelper   = AdaBoostHelper(self.dataHelper)
        scores, confusionMatrix = self.RunClassifier(parameters)
        self.assertAlmostEqual(confusionMatrix[1, 1], 25)


    @unittest.skipIf("Gradient Boosting" in skippedTests, "Skipped gradient boosting unit test.")
    def testGradientBoostingClassifier(self):
        parameters = {"n_estimators" : [100, 150, 200, 250],
                      "subsample"    : [0.8, 0.9, 1],
                      "max_features" : [0.7, 0.8, 0.9, 1]}

        self.regressionHelper   = GradientBoostingHelper(self.dataHelper, )
        scores, confusionMatrix = self.RunClassifier(parameters)
        self.assertAlmostEqual(confusionMatrix[1, 1], 47)


    @unittest.skipIf("X Gradient Boosting" in skippedTests, "Skipped extreme gradient boosting unit test.")
    def testXGradientBoostingClassifier(self):
        parameters = {"n_estimators": np.arange(10, 20),
                      "scale_pos_weight"  : [0, 5],
                      "subsample"         : [0.5, 0.9],
                      "learning_rate"     : [0.2, 0.05],
                      "gamma"             : [0, 3],
                      "colsample_bytree"  : [0.5, 0.9],
                      "colsample_bylevel" : [0.5, 0.9]}

        self.regressionHelper   = XGradientBoostingHelper(self.dataHelper)
        scores, confusionMatrix = self.RunClassifier(parameters)
        self.assertAlmostEqual(confusionMatrix[1, 1], 81)


    def RunClassifier(self, parameters):
        self.regressionHelper.SplitData(TestHyperparameterHelper.dependentVariable, 0.3)
        self.hyperparameterHelper   = HyperparameterHelper(self.regressionHelper)

        self.regressionHelper.CreateModel()
        self.hyperparameterHelper.CreateGridSearchModel(parameters, metrics.recall_score)
        self.regressionHelper.Predict()

        self.regressionHelper.dataHelper.consoleHelper.Print("\n", ConsoleHelper.VERBOSEREQUESTED)
        self.regressionHelper.PrintClassName()
        self.hyperparameterHelper.DisplayChosenParameters()

        self.regressionHelper.DisplayModelPerformanceScores()
        scores = self.regressionHelper.GetModelPerformanceScores()

        self.regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")
        confusionMatrix = self.regressionHelper.GetConfusionMatrix(dataSet="testing")
        return scores, confusionMatrix


if __name__ == "__main__":
    unittest.main()