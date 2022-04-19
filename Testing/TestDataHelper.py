# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
import DataSetLoading

from lendres.ConsoleHelper import ConsoleHelper

import unittest

class TestDataHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        verboseLevel = ConsoleHelper.VERBOSEREQUESTED

        cls.insuranceDataHelper, cls.insuranceDependentVariable = DataSetLoading.GetInsuranceData(verboseLevel=ConsoleHelper.VERBOSEREQUESTED, encode=False)

        cls.loanData, cls.loanDependentVariable = DataSetLoading.GetLoanModellingData(verboseLevel=verboseLevel, dropExtra=False)
        cls.loanData.ChangeToCategoryType(["CreditCard", "Online"])

        cls.dataWithErrors, dependentVariable   = DataSetLoading.GetDataWithErrors(verboseLevel=verboseLevel)

        cls.usedCarData, dependentVariable      = DataSetLoading.GetUsedCarsData(verboseLevel=verboseLevel)

        cls.boundaries      = [0,     90000,   91000,   92000,   93000,   94000,   95000,   96000,   99999]
        cls.labels          = ["Os", "90000", "91000", "92000", "93000", "94000", "95000", "96000", "99999"]


    def setUp(self):
        self.insuranceDataHelper = TestDataHelper.insuranceDataHelper.Copy(deep=True)
        self.loanData            = TestDataHelper.loanData.Copy(deep=True)
        self.dataWithErrors      = TestDataHelper.dataWithErrors.Copy(deep=True)
        self.usedCarData         = TestDataHelper.usedCarData.Copy(deep=True)


    def testValueCounts(self):
        newColumnName = self.loanData.MergeNumericalDataByRange("ZIPCode", TestDataHelper.labels, TestDataHelper.boundaries);
        self.assertEqual(self.loanData.data[newColumnName].value_counts()["96000"], 40)


    def testNotAvailableCounts(self):
        # Test getting the not available counts with data missing.
        notAvailableCounts, totalNotAvailable = self.dataWithErrors.GetNotAvailableCounts()
        self.assertEqual(totalNotAvailable, 1)

        # Remove the missing data and recheck to make sure it was removed.
        self.dataWithErrors.DropRowsWhereDataNotAvailable(["children"])
        notAvailableCounts, totalNotAvailable = self.dataWithErrors.GetNotAvailableCounts()
        self.assertEqual(totalNotAvailable, 0)


    def testGetMinAndMaxValues(self):
        result = TestDataHelper.loanData.GetMinAndMaxValues("Income", 5, method="quantity")
        self.assertEqual(result["Largest"].iloc[-1], 224)

        solution = TestDataHelper.loanData.data.shape[0] * 0.05
        result   = TestDataHelper.loanData.GetMinAndMaxValues("Income", 5, method="percent")
        self.assertAlmostEqual(len(result["Largest"]), solution, 0)


    def testDisplaying(self):
        self.loanData.DisplayAllCategoriesValueCounts()
        self.loanData.DisplayUniqueValues(["Online", "CreditCard"])


    def testStringExtraction(self):
        columns = ["Mileage", "Engine", "Power"]
        # For this data, the not available rows need to be removed.
        self.usedCarData.DropAllRowsWhereDataNotAvailable()

        result = self.usedCarData.ExtractLastStringTokens(columns)
        result = result.nunique()

        self.usedCarData.consoleHelper.PrintTitle("Extracted String Token Counts", ConsoleHelper.VERBOSEREQUESTED)
        self.usedCarData.consoleHelper.Display(result, ConsoleHelper.VERBOSEREQUESTED)

        self.assertEqual(result.loc["Mileage"], 2)
        self.assertEqual(result.loc["Engine"], 1)


    def testCategoryConversion(self):
        self.insuranceDataHelper.data["smoker"] = self.insuranceDataHelper.data["smoker"].astype("category")
        self.insuranceDataHelper.ConvertCategoryToNumeric("smoker", "yes")


if __name__ == "__main__":
    unittest.main()