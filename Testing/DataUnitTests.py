# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
from lendres.DataHelper import DataHelper
from lendres.ConsoleHelper import ConsoleHelper
import unittest

class TestDataHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        consoleHelper       = ConsoleHelper(verboseLevel=ConsoleHelper.VERBOSENONE)
        cls.loanData        = DataHelper(consoleHelper=consoleHelper)
        cls.loanData.LoadAndInspectData("Loan_Modelling.csv")

        cls.dataWithErrors  = DataHelper(consoleHelper=consoleHelper)
        cls.dataWithErrors.LoadAndInspectData("datawitherrors.csv")

        cls.boundaries = [0,     90000,   91000,   92000,   93000,   94000,   95000,   96000,   99999]
        cls.labels     = ["Os", "90000", "91000", "92000", "93000", "94000", "95000", "96000", "99999"]


    def setUp(self):
        self.loanData = TestDataHelper.loanData.Copy(deep=True)


    def testValueCounts(self):
        newColumnName = self.loanData.MergeNumericalDataByRange("ZIPCode", TestDataHelper.labels, TestDataHelper.boundaries);
        self.assertEqual(self.loanData.data[newColumnName].value_counts()["96000"], 40)


    def testGetNotAvailableCounts(self):
        notAvailableCounts, totalNotAvailable = TestDataHelper.dataWithErrors.GetNotAvailableCounts()
        self.assertEqual(totalNotAvailable, 1)


    def testGetMinAndMaxValues(self):
        result = TestDataHelper.loanData.GetMinAndMaxValues("Income", 5, method="quantity")
        #print(result)
        self.assertEqual(result["Largest"].iloc[-1], 224)

        solution = TestDataHelper.loanData.data.shape[0] * 0.05
        result = TestDataHelper.loanData.GetMinAndMaxValues("Income", 5, method="percent")
        self.assertEqual(len(result["Largest"]), solution)


if __name__ == "__main__":
    unittest.main()