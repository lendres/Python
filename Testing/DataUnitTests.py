# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
import pandas as pd

import lendres
from lendres.Data import MergeNumericalDataByRange
import unittest


data = lendres.Data.LoadAndInspectData("data.csv")
print("\n\n\n")
data = lendres.Data.LoadAndInspectData("datawitherrors.csv")


class TestMergeNumericalDataByRange(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inputFile   = "Loan_Modelling.csv"
        #self.data   = lendres.Data.LoadAndInspectData(inputFile)
        self.loanData  = DataHelper()
        self.loanData.LoadAndInspect(inputFile)

        cls.data   = pd.read_csv(inputFile)
        
        cls.boundaries = [0,     90000,   91000,   92000,   93000,   94000,   95000,   96000,   99999]
        cls.labels     = ["Os", "90000", "91000", "92000", "93000", "94000", "95000", "96000", "99999"]
        
    def setUp(self):
        self.data = TestMergeNumericalDataByRange.data.copy(deep=True)
        
    def testValueCounts(self):
        newColumnName = MergeNumericalDataByRange(self.data, "ZIPCode", TestMergeNumericalDataByRange.labels, TestMergeNumericalDataByRange.boundaries);
        self.assertEqual(self.data[newColumnName].value_counts()["96000"], 40)

    def testGetNotAvailableCounts(self):
        notAvailableCounts, totalNotAvailable = lendres.Data.GetNotAvailableCounts(self.data)


if __name__ == "__main__":
    unittest.main()