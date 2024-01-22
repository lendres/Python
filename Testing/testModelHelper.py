"""
Created on January 26, 2022
@author: Lance A. Endres
"""
import DataSetLoading
from   lendres.io.ConsoleHelper                                      import ConsoleHelper
from   lendres.ModelHelper                                           import ModelHelper

from   lendres.BaggingHelper                                         import BaggingHelper
from   imblearn.over_sampling                                        import SMOTE

import unittest


class TestModelHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        verboseLevel = ConsoleHelper.VERBOSEREQUESTED
        verboseLevel = ConsoleHelper.VERBOSETESTING
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCardiacData(verboseLevel=verboseLevel)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """


    def testBasicSplit(self):
        dataHelper = self.dataHelper.Copy()

        modelHelper = ModelHelper(dataHelper, BaggingHelper(dataHelper))
        modelHelper.dataHelper.SplitData(self.dependentVariable, 0.3, stratify=False)

        result = modelHelper.dataHelper.GetSplitComparisons()
        self.dataHelper.consoleHelper.PrintNewLine(1, ConsoleHelper.VERBOSEREQUESTED)
        self.dataHelper.consoleHelper.Print(result, ConsoleHelper.VERBOSEREQUESTED)

        modelHelper = ModelHelper(dataHelper, BaggingHelper(dataHelper))
        modelHelper.dataHelper.SplitData(self.dependentVariable, 0.3, stratify=True)

        result = modelHelper.dataHelper.GetSplitComparisons()
        self.dataHelper.consoleHelper.PrintNewLine(1, ConsoleHelper.VERBOSEREQUESTED)
        self.dataHelper.consoleHelper.Print(result, ConsoleHelper.VERBOSEREQUESTED)


    def testValidationSplit(self):
        dataHelper = self.dataHelper.Copy()

        modelHelper = ModelHelper(dataHelper,  BaggingHelper(dataHelper))
        dataHelper.SplitData(self.dependentVariable, 0.2, 0.3, stratify=False)

        result = modelHelper.dataHelper.GetSplitComparisons()
        self.dataHelper.consoleHelper.PrintNewLine(1, ConsoleHelper.VERBOSEREQUESTED)
        self.dataHelper.consoleHelper.Print(result, ConsoleHelper.VERBOSEREQUESTED)

        regressionHelper = BaggingHelper(dataHelper)
        regressionHelper.dataHelper.SplitData(self.dependentVariable, 0.2, validationSize=0.25, stratify=True)

        sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)
        regressionHelper.dataHelper.xTrainingData, regressionHelper.dataHelper.yTrainingData = sm.fit_resample(regressionHelper.dataHelper.xTrainingData, regressionHelper.dataHelper.yTrainingData)

        regressionHelper.FitPredict()

        result = regressionHelper.GetModelPerformanceScores()
        self.dataHelper.consoleHelper.Display(result)

        self.assertAlmostEqual(result["Recall"]["Validation"], 0.5789, places=3)
        self.assertAlmostEqual(result["Precision"]["Validation"], 0.2650, places=3)


if __name__ == "__main__":
    unittest.main()