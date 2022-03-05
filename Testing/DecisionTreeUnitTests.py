# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
import pandas as pd
#from IPython.display import display

from lendres.ConsoleHelper import ConsoleHelper
from lendres.DataHelper import DataHelper
from lendres.DecisionTreeHelper import DecisionTreeHelper
import unittest

class TestDecisionTreeHelper(unittest.TestCase):

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
        consoleHelper   = ConsoleHelper(verboseLevel=ConsoleHelper.VERBOSENONE)
        cls.dataHelper  = DataHelper(consoleHelper=consoleHelper)
        cls.dataHelper.LoadAndInspectData(inputFile)
        cls.dependentVariable = dependentVariable

        if cls.whichData == 0:
            cls.fixLoanData()
        elif cls.whichData == 1:
            cls.fixCreditData()

        #print("\nData size after cleaning:")
        #display(cls.dataHelper.data.shape)


    @classmethod
    def fixLoanData(cls):
        cls.dataHelper.data.drop(["ID"], axis=1, inplace=True)
        cls.dataHelper.data.drop(["ZIPCode"], axis=1, inplace=True)
        cls.dataHelper.RemoveRowsWithValueOutsideOfCriteria("Experience", 0, "dropbelow", inPlace=True)
        cls.dataHelper.EncodeCategoricalColumns(["Family", "Education"])
        cls.dataHelper.DropOutliers("Income", inPlace=True)
        cls.dataHelper.DropOutliers("CCAvg", inPlace=True)
        cls.dataHelper.DropOutliers("Mortgage", inPlace=True)


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

        cls.dataHelper.ChangeAllObjectColumnsToCategories()
        cls.dataHelper.data = cls.dataHelper.data.replace(replaceStruct)
        cls.dataHelper.EncodeCategoricalColumns(columns=oneHotCols)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper         = TestDecisionTreeHelper.dataHelper.Copy(deep=True)
        self.regressionHelper   = DecisionTreeHelper(self.dataHelper)

        self.regressionHelper.SplitData(TestDecisionTreeHelper.dependentVariable, 0.3)


    def testStandardPlots(self):
        self.regressionHelper.CreateModel()
        self.regressionHelper.CreateDecisionTreePlot()
        self.regressionHelper.CreateFeatureImportancePlot()


    def testGetDependentVariableName(self):
        result = self.regressionHelper.GetDependentVariableName()
        self.assertEqual(result, TestDecisionTreeHelper.dependentVariable)


if __name__ == "__main__":
    unittest.main()