"""
Created on December 27, 2021
@author: Lance A. Endres
"""
import pandas                                    as pd
import matplotlib.pyplot                         as plt
import seaborn                                   as sns

import os

import DataSetLoading
from   lendres.plotting.PlotHelper               import PlotHelper
from   lendres.plotting.AxesHelper               import AxesHelper

import unittest


# By default this should be True.  It can be toggled to false if you want to see the
# output for the file saving tests (they won't be deleted).  Be advised, if you set this
# to True, you should perform file clean up operations afterwards.  You can manually delete
# the files, or set this back to True and rerun the tests.
deleteOutput = True


class TestPlotHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inputFile = "insurance.csv"
        inputFile = DataSetLoading.GetFileInDataDirectory(inputFile)
        cls.data  = pd.read_csv(inputFile)


    def testFormatPlotMethod1(self):
        self.createBasicPlot("Format by Scale", scale=2.0)
        plt.show()


    def testFormatPlotMethod2(self):
        self.createBasicPlot("Format by Width and Height", width=5, height=3)
        plt.show()


    def testAlternatPlotFormats(self):
        self.createBasicPlot("Format with Defaults", formatStyle=None, width=5, height=3)
        plt.show()
        self.createBasicPlot("Format with Seaborn", formatStyle="seaborn", width=5, height=3)
        plt.show()
        self.createBasicPlot("Format by Resetting Pyplot", formatStyle="pyplot", width=5, height=3)
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


    def createBasicPlot(self, titlePrefix, formatStyle=None, scale=1.0, width=10, height=6):
        PlotHelper.scale = scale
        PlotHelper.FormatPlot(formatStyle=formatStyle, width=width, height=height)
        axis = plt.gca()
        sns.histplot(TestPlotHelper.data["bmi"], kde=True, ax=axis, palette="winter")
        AxesHelper.Label(axis, title="Test Plot", xLabel="BMI", yLabel="Count", titlePrefix=titlePrefix)

        # Reset the scale to the default for the next plot.
        PlotHelper.scale = 1.0
        return plt.gcf()


    def getFullPath(self, fileName):
        return os.path.join(PlotHelper.GetDefaultOutputDirectory(), fileName)


    @classmethod
    def tearDownClass(cls):
        # It's not known what test function will be last, so make sure we clean
        # up any files and directories created.
        if deleteOutput:
            PlotHelper.DeleteOutputDirectory()


if __name__ == "__main__":
    unittest.main()