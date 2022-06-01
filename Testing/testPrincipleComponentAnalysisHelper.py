"""
Created on April 27, 2022
@author: Lance A. Endres
"""
import pandas                                     as pd

import DataSetLoading

from   lendres.ConsoleHelper                      import ConsoleHelper
from   lendres.PrincipleComponentAnalysisHelper   import PrincipleComponentAnalysisHelper

from   IPython.display                            import display

import unittest

class TestPrincipleComponentAnalysisHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        verboseLevel = ConsoleHelper.VERBOSEREQUESTED

        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCarMpgData(verboseLevel=verboseLevel)
        # Used to display all the columns in the output.
        pd.set_option("display.max_columns", None)

    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper                       = TestPrincipleComponentAnalysisHelper.dataHelper.Copy(deep=True)
        self.principleComponentAnalysisHelper = PrincipleComponentAnalysisHelper(self.dataHelper, ["mpg", "car name", "origin"], copyMethod="exclude")
        self.principleComponentAnalysisHelper.ScaleData(method="zscore")


    def testFit(self):
        self.principleComponentAnalysisHelper.CreateModel()
        self.principleComponentAnalysisHelper.Fit()
        eigenvalues = self.principleComponentAnalysisHelper.model.explained_variance_ratio_
        self.assertAlmostEqual(eigenvalues[0], 0.7088, places=3)


    def testTransform(self):
        self.principleComponentAnalysisHelper.CreateModel(3)
        reducedData = self.principleComponentAnalysisHelper.FitTransform()
        eigenvalues = self.principleComponentAnalysisHelper.model.explained_variance_ratio_
        self.assertAlmostEqual(eigenvalues[0], 0.7088, places=3)
        self.principleComponentAnalysisHelper.CreateVarianceExplainedPlot()


    def testCreateVarianceExplainedPlot(self):
        self.principleComponentAnalysisHelper.CreateModel()
        self.principleComponentAnalysisHelper.Fit()
        self.principleComponentAnalysisHelper.CreateVarianceExplainedPlot()


if __name__ == "__main__":
    unittest.main()