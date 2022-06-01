"""
Created on January 26, 2022
@author: Lance A. Endres
"""
from IPython.display import display

import DataSetLoading
from lendres.DataHelper              import DataHelper
from lendres.ModelHelper             import ModelHelper

from lendres.BaggingHelper           import BaggingHelper
from imblearn.over_sampling          import SMOTE

import unittest


class TestModelHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCardiacData()


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """


    def testBasicSplit(self):
        dataHelper = DataHelper.Copy(TestModelHelper.dataHelper, deep=True)

        modelHelper = ModelHelper(dataHelper, BaggingHelper(dataHelper))
        modelHelper.dataHelper.SplitData(TestModelHelper.dependentVariable, 0.3, stratify=False)

        result = modelHelper.dataHelper.GetSplitComparisons()
        print()
        display(result)

        modelHelper = ModelHelper(dataHelper, BaggingHelper(dataHelper))
        modelHelper.dataHelper.SplitData(TestModelHelper.dependentVariable, 0.3, stratify=True)

        result = modelHelper.dataHelper.GetSplitComparisons()
        print()
        display(result)


    def testValidationSplit(self):
        dataHelper = DataHelper.Copy(TestModelHelper.dataHelper, deep=True)

        modelHelper = ModelHelper(dataHelper,  BaggingHelper(dataHelper))
        dataHelper.SplitData(TestModelHelper.dependentVariable, 0.2, 0.3, stratify=False)

        result = modelHelper.dataHelper.GetSplitComparisons()
        print()
        display(result)

        regressionHelper = BaggingHelper(dataHelper)
        regressionHelper.dataHelper.SplitData(TestModelHelper.dependentVariable, 0.2, validationSize=0.25, stratify=True)

        sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)
        regressionHelper.dataHelper.xTrainingData, regressionHelper.dataHelper.yTrainingData = sm.fit_resample(regressionHelper.dataHelper.xTrainingData, regressionHelper.dataHelper.yTrainingData)

        regressionHelper.FitPredict()

        result = regressionHelper.GetModelPerformanceScores()
        #display(result)

        self.assertAlmostEqual(result["Recall"]["Validation"], 0.5789, places=3)
        self.assertAlmostEqual(result["Precision"]["Validation"], 0.2650, places=3)


if __name__ == "__main__":
    unittest.main()