# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""

import lendres
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
        self.loanData = DataHelper.Copy(TestDataHelper.loanData, deep=True)


    def testValueCounts(self):
        newColumnName = self.loanData.MergeNumericalDataByRange("ZIPCode", TestDataHelper.labels, TestDataHelper.boundaries);
        self.assertEqual(self.loanData.data[newColumnName].value_counts()["96000"], 40)


    def testGetNotAvailableCounts(self):
        notAvailableCounts, totalNotAvailable = TestDataHelper.dataWithErrors.GetNotAvailableCounts()


if __name__ == "__main__":
    unittest.main()