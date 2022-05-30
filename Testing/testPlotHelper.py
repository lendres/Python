"""
Created on December 27, 2021
@author: Lance
"""
import seaborn                 as sns
import matplotlib.pyplot       as plt
import pandas                  as pd

import os

from lendres.PlotHelper        import PlotHelper

import unittest

class TestPlotHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inputFile = "insurance.csv"
        inputFile = os.path.join("Data", inputFile)
        cls.data   = pd.read_csv(inputFile)


    def testFormatPlotMethod1(self):
        self.createBasicPlot("Format by Scale", scale=2.0)
        plt.show()


    def testFormatPlotMethod2(self):
        self.createBasicPlot("Format by Width and Height", width=5, height=3)
        plt.show()


    def testSavePlotBeforeShowMethod1(self):
        self.createBasicPlot("Save Before Show 1")

        # Test with current figure.
        fileName = "Plot Before Show (gcf).png"
        PlotHelper.SavePlot(fileName)

        fullPath = self.getFullPath(fileName)
        self.assertTrue(os.path.exists(fullPath))
        plt.show()


    def testSavePlotBeforeShowMethod2(self):
        figure = self.createBasicPlot("Save Before Show 2")

        # Test with supplied figure.
        fileName = "Plot Before Show (figure).png"
        PlotHelper.SavePlot(fileName, figure=figure)

        fullPath = self.getFullPath(fileName)
        self.assertTrue(os.path.exists(fullPath))
        plt.show()


    def createBasicPlot(self, titlePrefix, scale=1.0, width=10, height=6):
        PlotHelper.scale = scale
        PlotHelper.FormatPlot(width=width, height=height)
        axis = plt.gca()
        sns.histplot(TestPlotHelper.data["bmi"], kde=True, ax=axis, palette="winter")
        PlotHelper.Label(axis, title="Test Plot", xLabel="BMI", yLabel="Count", titlePrefix=titlePrefix)

        # Reset the scale to the default for the next plot.
        PlotHelper.scale = 1.0
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