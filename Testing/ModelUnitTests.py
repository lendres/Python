# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:53:03 2022

@author: Lance
"""
from IPython.display import display

import DataSetLoading
from lendres.DataHelper import DataHelper
from lendres.ModelHelper import ModelHelper

import unittest


class TestModelHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCreditData()


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """


    def testBasicSplit(self):
        dataHelper = DataHelper.Copy(TestModelHelper.dataHelper, deep=True)

        modelHelper = ModelHelper(dataHelper)
        modelHelper.SplitData(TestModelHelper.dependentVariable, 0.3, stratify=False)

        result = modelHelper.GetSplitComparisons()
        display(result)

        #self.assertAlmostEqual(result["Coefficients"]["age"], 251.681865, places=3)

    def testValidationSplit(self):
        dataHelper = DataHelper.Copy(TestModelHelper.dataHelper, deep=True)

        modelHelper = ModelHelper(dataHelper)
        modelHelper.SplitData(TestModelHelper.dependentVariable, 0.2, 0.3, stratify=False)

        result = modelHelper.GetSplitComparisons()
        display(result)


if __name__ == "__main__":
    unittest.main()