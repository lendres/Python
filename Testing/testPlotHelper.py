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


    # @unittest.skip
    def testArtisticPlot(self):
        PlotHelper.FormatNewArtisticFigure()
        plt.show()


    # @unittest.skip
    def testBuiltInStyleFormats(self):
        self.CreateBasicPlot("Built In - Format with Defaults")

        # Test using the file extension or not using the file extension.
        PlotHelper.PushSettings(parameterFile="ggplot")
        self.CreateBasicPlot("Built In - ggplot")
        PlotHelper.PopSettings()


    # @unittest.skip
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


    # @unittest.skip
    def testCopySettings(self):
        self.CreateBasicPlot("Settings - Default Formatting")
        PlotHelper.PushSettings(scale=2.0)
        self.CreateBasicPlot("Settings - Initial Format Settings")

        settings = PlotHelper.FormatSettings.Copy()
        settings.ParameterFile = "seaborn"
        PlotHelper.PushSettings(settings)
        self.CreateBasicPlot("Settings - Copied Format Settings")

        PlotHelper.PopSettings()
        self.CreateBasicPlot("Settings - Popped Format Settings")


    def testExceptionForStyleFile(self):
        # Test the exception.
        PlotHelper.PushSettings(parameterFile="invalid")
        self.assertRaises(Exception, self.CreateBasicPlot, "Test Exception")
        PlotHelper.PopSettings()


    def testFindInCurrentDirectory(self):
        # Test finding in the current directory.  Formatting should be a little different.
        PlotHelper.PushSettings(parameterFile="test")
        self.CreateBasicPlot("Test.mplstyle")
        PlotHelper.PopSettings()


    # @unittest.skip
    def testNumberFormatException(self):
        # Should not cause an exception.
        PlotHelper.GetColorCycle(numberFormat="RGB")
        PlotHelper.GetColorCycle(lineColorCycle="seaborn", numberFormat="hex")

        # Test the exception.
        self.assertRaises(Exception, PlotHelper.GetColorCycle, numberFormat="invalid")


    # @unittest.skip
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


    # @unittest.skip
    def testPlotAllStyles(self):
        styleFiles = PlotHelper.GetListOfPlotStyles()
        for styleFile in styleFiles:
            PlotHelper.PushSettings(parameterFile=styleFile)
            self.CreateBasicPlot("Plot Styles - Format with "+styleFile)
        PlotHelper.PopSettings()


    # @unittest.skip
    def testPushIndividualtSettings(self):
        self.CreateBasicPlot("Individual Settings - Format with Defaults")
        PlotHelper.PushSettings(scale=2.0)
        self.CreateBasicPlot("Individual Settings - Pushed Format Settings")
        PlotHelper.PopSettings()
        self.CreateBasicPlot("Individual Settings - Popped Format Settings")


    # @unittest.skip
    def testSavePlotBeforeShowMethod1(self):
        self.CreateBasicPlot("Save Figure")

        # Test with current figure.
        fileName = "Test Plot.png"
        PlotHelper.SavePlot(fileName)

        fullPath = self.GetFullPath(fileName)
        self.assertTrue(os.path.exists(fullPath))


    # @unittest.skip
    def testScaleVersusParameterFiles(self):
        self.CreateBasicPlot("Scale and File - Format with Defaults")
        PlotHelper.PushSettings(scale=0.8)
        self.CreateBasicPlot("Scale and File - Formated with Scale")
        PlotHelper.PushSettings(parameterFile="mediumsizefont")
        self.CreateBasicPlot("Scale and File - Formated with File")
        PlotHelper.PopSettings()


    # @unittest.skip
    def testSetPushPopSettings1(self):
        self.CreateBasicPlot("Set Push Pop 1 - Format with Defaults")
        PlotHelper.SetSettings(overrides={"figure.figsize" : (8, 8), "axes.titlesize" : 15})
        self.CreateBasicPlot("Set Push Pop 1 - Formated with Set Settings")
        PlotHelper.PushSettings(overrides={"axes.titlesize" : 22})
        self.CreateBasicPlot("Set Push Pop 1 - Pushed Settings")
        PlotHelper.PopSettings()
        self.CreateBasicPlot("Set Push Pop 1 - Popped Settings")
        PlotHelper.ResetSettings()
        self.CreateBasicPlot("Set Push Pop 1 - Reset Settings")


    # @unittest.skip
    def testSetPushPopSettings2(self):
        self.CreateBasicPlot("Set Push Pop 2 - Format with Defaults")
        PlotHelper.PushSettings(formatSettings="default", overrides={"figure.figsize" : (8, 8), "axes.titlesize" : 15})
        self.CreateBasicPlot("Set Push Pop 2 - Formated with Push Settings on Defaults")
        PlotHelper.PushSettings(FormatSettings(overrides={"axes.grid" : False}))
        self.CreateBasicPlot("Set Push Pop 2 - Pushed FormatSettings")
        PlotHelper.PopSettings()
        self.CreateBasicPlot("Set Push Pop 2 - Popped Settings")

        # Test the exception.
        self.assertRaises(Exception, PlotHelper.PushSettings, formatSettings="invalid")


    def CreateBasicPlot(self, title):
        PlotHelper.Format()

        figure = plt.gcf()
        axes   = plt.gca()
        sns.histplot(self.data["bmi"], kde=True, ax=axes, label="Data")
        AxesHelper.Label(axes, title=title, xLabels="Values", yLabels="Count")

        axes.legend()
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