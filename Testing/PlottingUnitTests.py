# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:30:11 2021

@author: Lance
"""

# Use this to import from another directory.
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import os

from lendres.PlotHelper import PlotHelper

import unittest

class TestPlotting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inputFile = "insurance.csv"
        inputFile = os.path.join("Data", inputFile)
        cls.data   = pd.read_csv(inputFile)


    def testFormatPlotMethod1(self):
        self.createBasicPlot(scale=2.0)
        plt.show()


    def testFormatPlotMethod2(self):
        self.createBasicPlot(width=5, height=3)
        plt.show()


    def testSavePlotBeforeShowMethod1(self):
        self.createBasicPlot()

        # Test with current figure.
        fileName = "Plot Before Show (gcf).png"
        PlotHelper.SavePlot(fileName, useDefaultOutputFolder=True)

        fullPath = self.getFullPath(fileName)
        self.assertTrue(os.path.exists(fullPath))
        plt.show()


    def testSavePlotBeforeShowMethod2(self):
        figure = self.createBasicPlot()

        # Test with supplied figure.
        fileName = "Plot Before Show (figure).png"
        PlotHelper.SavePlot(fileName, figure=figure, useDefaultOutputFolder=True)

        fullPath = self.getFullPath(fileName)
        self.assertTrue(os.path.exists(fullPath))
        plt.show()


    def createBasicPlot(self, scale=1.0, width=10, height=6):
        PlotHelper.FormatPlot(scale=scale, width=width, height=height)
        axis = plt.gca()
        sns.histplot(TestPlotting.data["bmi"], kde=True, ax=axis, palette="winter")
        axis.set(title="Test Plot", xlabel="BMI", ylabel="Count")
        return plt.gcf()


    def getFullPath(self, fileName):
        return os.path.join(PlotHelper.GetDefaultOutputDirectory(), fileName)


    @classmethod
    def tearDownClass(cls):
        # It's not known what test function will be last, so make sure we clean
        # up any files and directories created.
        PlotHelper.DeleteOutputDirectory()


if __name__ == "__main__":
    unittest.main()