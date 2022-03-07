# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""
import os

import lendres

from lendres.DataHelper import DataHelper
from lendres.ConsoleHelper import ConsoleHelper

import unittest

class TestUnivariateAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inputFile           = "data.csv"

        inputFile           = os.path.join("Data", inputFile)

        consoleHelper       = ConsoleHelper(verboseLevel=ConsoleHelper.VERBOSENONE)

        cls.dataHelper      = DataHelper(consoleHelper=consoleHelper)
        cls.dataHelper.LoadAndInspectData(inputFile)


    def setUp(self):
        self.dataHelper = TestUnivariateAnalysis.dataHelper.Copy(deep=True)


    def testPlots(self):
        saveImages = False

        categories = ["bmi", "charges"]
        lendres.Plotting.ApplyPlotToEachCategory(lendres.UnivariateAnalysis.CreateBoxAndHistogramPlot, self.dataHelper.data, categories, save=saveImages)

        categories = ["children", "smoker"]
        lendres.Plotting.ApplyPlotToEachCategory(lendres.UnivariateAnalysis.CreatePercentageBarPlot, self.dataHelper.data, categories, save=saveImages)

        lendres.UnivariateAnalysis.CreateBoxPlot(self.dataHelper.data, "charges")


if __name__ == "__main__":
    unittest.main()