# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.metrics import recall_score

import lendres
from lendres.DecisionTreeHyperparameterHelper import DecisionTreeHyperparameterHelper
import unittest

class TestDecisionTreeHyperparameterHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.whichData = 1

        inputFile                 = ""
        dependentVariable         = ""

        if cls.whichData == 0:
            inputFile             = "Loan_Modelling.csv"
            dependentVariable     = "Personal_Loan"
        elif cls.whichData == 1:
            inputFile             = "credit.csv"
            dependentVariable     = "default"
        else:
            raise Exception("Input selection is incorrect.")

        #cls.data = lendres.Data.LoadAndInspectData(inputFile)
        cls.data              = pd.read_csv(inputFile)
        cls.dependentVariable = dependentVariable

        if cls.whichData == 0:
            cls.fixLoanData()
        elif cls.whichData == 1:
            cls.fixCreditData()

        print("\nData size after cleaning:")
        display(cls.data.shape)


    @classmethod
    def fixLoanData(cls):
        cls.data.drop(["ID"], axis=1, inplace=True)
        cls.data.drop(["ZIPCode"], axis=1, inplace=True)
        lendres.Data.RemoveRowsWithValueOutsideOfCriteria(cls.data, "Experience", 0, "dropbelow", inPlace=True)
        lendres.Data.EncodeCategoricalColumns(cls.data, ["Family", "Education"])
        lendres.Data.DropOutliers(cls.data, "Income", inPlace=True)
        lendres.Data.DropOutliers(cls.data, "CCAvg", inPlace=True)
        lendres.Data.DropOutliers(cls.data, "Mortgage", inPlace=True)


    @classmethod
    def fixCreditData(cls):
        replaceStruct = {"checking_balance"    : {"< 0 DM" : 1, "1 - 200 DM" : 2,"> 200 DM" : 3, "unknown" : -1},
                         "credit_history"      : {"critical" : 1, "poor" : 2, "good" : 3, "very good" : 4, "perfect" : 5},
                         "savings_balance"     : {"< 100 DM" : 1, "100 - 500 DM" : 2, "500 - 1000 DM" : 3, "> 1000 DM" : 4, "unknown" : -1},
                         "employment_duration" : {"unemployed" : 1, "< 1 year" : 2, "1 - 4 years" : 3, "4 - 7 years" : 4, "> 7 years" : 5},
                         "phone"               : {"no" : 1, "yes" : 2 },
                         "default"             : {"no" : 0, "yes" : 1 }
                         }
        oneHotCols = ["purpose", "housing", "other_credit", "job"]

        # Loop through all columns in the dataframe.
        for feature in cls.data.columns:
            # Only apply for columns with categorical strings.
            if cls.data[feature].dtype == 'object':
                # Replace strings with an integer.
                cls.data[feature] = pd.Categorical(cls.data[feature])

        cls.data = cls.data.replace(replaceStruct)
        cls.data = pd.get_dummies(cls.data, columns=oneHotCols)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.data             = TestDecisionTreeHyperparameterHelper.data.copy(deep=True)
        self.regressionHelper = DecisionTreeHyperparameterHelper(self.data)

        self.regressionHelper.SplitData(TestDecisionTreeHyperparameterHelper.dependentVariable, 0.3)
   

    def testHyperparameterTuning(self):
        parameters = {"max_depth"             : np.arange(1, 3),
                      "min_samples_leaf"      : [2, 5, 7],
                      "max_leaf_nodes"        : [2, 5, 10],
                      "criterion"             : ["entropy", "gini"]
                      }

        self.regressionHelper.CreateModel()
        self.regressionHelper.CreateGridSearchModel(parameters, scoringFunction=recall_score)


if __name__ == "__main__":
    unittest.main()