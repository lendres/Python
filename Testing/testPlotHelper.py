"""
Created on December 27, 2021
@author: Lance A. Endres
"""
import pandas                                                        as pd
import matplotlib.pyplot                                             as plt
import seaborn                                                       as sns

import os

import DataSetLoading
from   lendres.plotting.FormatSettings                               import FormatSettings
from   lendres.plotting.PlotHelper                                   import PlotHelper
from   lendres.plotting.AxesHelper                                   import AxesHelper

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


    def setUp(self):
        PlotHelper.ResetSettings()


    def testArtistiPlot(self):
        PlotHelper.NewArtisticFigure()
        plt.show()


    def testCompareSeabornToSeaborn(self):
        """
        Compare the real Seaborn style to the "seaborn.mplstyle" version.
        """
        sns.set(color_codes=True)
        #print(plt.rcParams)
        axis = plt.gca()
        sns.histplot(self.data["bmi"], kde=True, ax=axis)
        AxesHelper.Label(axis, title="Seaborn Comparison - Seaborn Generated", xLabels="Values", yLabels="Count")
        plt.show()

        PlotHelper.PushSettings(parameterFile="seaborn", scale=0.6)
        self.CreateBasicPlot("Seaborn Comparison - Using Parameter File")
        PlotHelper.PopSettings()


    def testCopySettings(self):
        self.CreateBasicPlot("Settings - Default Formatting")
        PlotHelper.PushSettings(scale=2.0)
        self.CreateBasicPlot("Settings - Initial Format Settings")

        settings = PlotHelper.GetSettings().Copy()
        settings.ParameterFile = "seaborn"
        PlotHelper.PushSettings(settings)
        self.CreateBasicPlot("Settings - Copied Format Settings")

        PlotHelper.PopSettings()
        self.CreateBasicPlot("Settings - Popped Format Settings")


    def testNumberFormatException(self):
        # Should not cause an exception.
        PlotHelper.GetColorCycle(numberFormat="RGB")
        PlotHelper.GetColorCycle(lineColorCycle="seaborn", numberFormat="hex")

        # Test the exception.
        self.assertRaises(Exception, PlotHelper.GetColorCycle, numberFormat="invalid")


    def testPlotStyleFormats(self):
        self.CreateBasicPlot("Style File Names - Format with Defaults")

        # Test using the file extension or not using the file extension.
        PlotHelper.PushSettings(parameterFile="gridless.mplstyle")
        self.CreateBasicPlot("Style File Names - With File Extension")
        PlotHelper.PushSettings(parameterFile="gridless")
        self.CreateBasicPlot("Style File Names - Without File Extension")
        PlotHelper.PopSettings()

        # Test that 2 pushes in a row did not lose original settings.
        self.CreateBasicPlot("Style File Names - Popped Format Settings")


    def testPlotAllStyles(self):
        styleFiles = PlotHelper.GetListOfPlotStyles()
        for styleFile in styleFiles:
            PlotHelper.PushSettings(parameterFile=styleFile)
            self.CreateBasicPlot("Plot Styles - Format with "+styleFile)
        PlotHelper.PopSettings()


    def testPushIndividualtSettings(self):
        self.CreateBasicPlot("Individual Settings - Format with Defaults")
        PlotHelper.PushSettings(scale=2.0)
        self.CreateBasicPlot("Individual Settings - Pushed Format Settings")
        PlotHelper.PopSettings()
        self.CreateBasicPlot("Individual Settings - Popped Format Settings")


    def testSavePlotBeforeShowMethod1(self):
        self.CreateBasicPlot("Save Figure")

        # Test with current figure.
        fileName = "Test Plot.png"
        PlotHelper.SavePlot(fileName)

        fullPath = self.GetFullPath(fileName)
        self.assertTrue(os.path.exists(fullPath))


    def testScaleVersusParameterFiles(self):
        self.CreateBasicPlot("Scale and File - Format with Defaults")
        PlotHelper.PushSettings(scale=0.8)
        self.CreateBasicPlot("Scale and File - Formated with Scale")
        PlotHelper.PushSettings(parameterFile="mediumsizefont")
        self.CreateBasicPlot("Scale and File - Formated with File")
        PlotHelper.PopSettings()


    def testSetPushPopSettings(self):
        self.CreateBasicPlot("Set Push Pop - Format with Defaults")
        PlotHelper.SetSettings(overrides={"figure.figsize" : (8, 8), "axes.titlesize" : 15})
        self.CreateBasicPlot("Set Push Pop - Formated with Set Settings")
        PlotHelper.PushSettings(overrides={"axes.titlesize" : 22})
        self.CreateBasicPlot("Set Push Pop - Pushed Settings")
        PlotHelper.PopSettings()
        self.CreateBasicPlot("Set Push Pop - Popped Settings")
        PlotHelper.ResetSettings()
        self.CreateBasicPlot("Set Push Pop - Reset Settings")


    def CreateBasicPlot(self, title):
        PlotHelper.Format()

        figure = plt.gcf()
        axis   = plt.gca()
        sns.histplot(self.data["bmi"], kde=True, ax=axis, label="Data")
        AxesHelper.Label(axis, title=title, xLabels="Values", yLabels="Count")

        axis.legend()
        plt.show()

        return figure


    def GetFullPath(self, fileName):
        return os.path.join(PlotHelper.GetDefaultOutputDirectory(), fileName)


    @classmethod
    def tearDownClass(cls):
        # It's not known what test function will be last, so make sure we clean
        # up any files and directories created.
        if deleteOutput:
            PlotHelper.DeleteOutputDirectory()


if __name__ == "__main__":
    unittest.main()