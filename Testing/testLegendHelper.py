"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import matplotlib.pyplot                                        as plt

from   lendres.plotting.AxesHelper                              import AxesHelper
from   lendres.plotting.PlotHelper                              import PlotHelper
from   lendres.plotting.PlotMaker                               import PlotMaker
from   lendres.plotting.LegendHelper                            import LegendHelper
from   lendres.plotting.LegendOptions                           import LegendOptions
from   lendres.demonstration.FunctionGenerator                  import FunctionGenerator

import unittest


class TestLegendHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # Generate a data set of 4 sine waves.
        cls.sinesDataFrame = FunctionGenerator.SineWavesAsDataFrame(magnitude=[10, 6, 8, 2], frequency=[4, 8, 2, 1], yOffset=[0, 22, 0, 2], slope=[10, 0, -6, 0], steps=1000)

        cls.sinesDataFrame.rename({"y0" : "Sine A1"}, axis="columns", inplace=True)
        cls.sinesDataFrame.rename({"y1" : "Sine B1"}, axis="columns", inplace=True)
        cls.sinesDataFrame.rename({"y2" : "Sine A2"}, axis="columns", inplace=True)
        cls.sinesDataFrame.rename({"y3" : "Sine B2"}, axis="columns", inplace=True)


    def testPlacementAndColumnsPlot(self):
        """
        Demonstrate multi-axeses plotting.
        """
        legendOptions = LegendOptions(location="outsidebottomleft", numberOfColumns=2)
        self.PlotMultiAxes("No Key Word Arguments", legendOptions)

        legendOptions = LegendOptions(location="outsidebottomcenter", numberOfColumns=2)
        self.PlotMultiAxes("No Key Word Arguments", legendOptions)

        legendOptions = LegendOptions(location="ousiderightcenter", numberOfColumns=1)
        self.PlotMultiAxes("No Key Word Arguments", legendOptions)


    def testLineWidthPlot(self):
        """
        Demonstrate multi-axeses plotting.
        """
        legendOptions = LegendOptions(lineWidth=None)
        self.PlotMultiAxes("No Key Word Arguments", legendOptions)

        legendOptions = LegendOptions(lineWidth=8)
        self.PlotMultiAxes("No Key Word Arguments", legendOptions)


    def PlotMultiAxes(self, titleSuffix, legendOptions, **kwargs):
        PlotHelper.Format()
        figure, axeses = PlotMaker.NewMultiYAxesPlot(self.sinesDataFrame, "x", [["Sine A1", "Sine A2"], ["Sine B1", "Sine B2"]], **kwargs)
        AxesHelper.Label(axeses, title="Multiple Y Axis Plot\n"+titleSuffix, xLabels="Time", yLabels=["Left (A)", "Right (B)"])
        LegendHelper.CreateLegend(figure, axeses[0], legendOptions)
        plt.show()


if __name__ == "__main__":
    unittest.main()