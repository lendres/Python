"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import pandas                                                   as pd
import numpy                                                    as np
import matplotlib.pyplot                                        as plt

import DataSetLoading
from   lendres.plotting.AxesHelper                              import AxesHelper
from   lendres.plotting.PlotMaker                               import PlotMaker
from   lendres.demonstration.FunctionGenerator                  import FunctionGenerator

import unittest


class TestPlotMaker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inputFile  = "insurance.csv"
        inputFile  = DataSetLoading.GetFileInDataDirectory(inputFile)
        cls.data   = pd.read_csv(inputFile)

        cls.confusionMatrix3x3 = np.array(
            [[25,  5,  10],
             [ 6, 10,   4],
             [ 8,  6,  15]]
        )


    def testConfusionMatrix(self):
        PlotMaker.CreateConfusionMatrixPlot(self.confusionMatrix3x3, "3 by 3 Confusion Matrix")
        PlotMaker.colorMap = "Blues"
        PlotMaker.CreateConfusionMatrixPlot(self.confusionMatrix3x3, "3 by 3 Confusion Matrix")
        PlotMaker.colorMap = None
        labels = ["car", "boat", "train"]
        PlotMaker.CreateConfusionMatrixPlot(self.confusionMatrix3x3, "3 by 3 Confusion Matrix", axesLabels=labels)


    def testPlotColorCycle(self):
        # A test that will also conveniently display the color cycles for reference.
        PlotMaker.PlotColorCycle(lineColorCycle="pyplot")
        PlotMaker.PlotColorCycle(lineColorCycle="seaborn")


    def testCreateFastFigure(self):
        x, a = FunctionGenerator.GetSineWave(magnitude=10, frequency=4, yOffset=0, slope=0, steps=1000)
        x, b = FunctionGenerator.GetSineWave(magnitude=4, frequency=2, yOffset=0, slope=10, steps=1000)
        x, c = FunctionGenerator.GetSineWave(magnitude=5, frequency=3, yOffset=30, slope=-5, steps=1000)
        PlotMaker.CreateFastFigure([a, b])
        PlotMaker.CreateFastFigure([a, b], yDataLabels=["Y 1", "Y 2"], xData=x, title="Test", xLabel="Time", yLabel="Value", linewidth=7)
        PlotMaker.CreateFastFigure([a, b], yDataLabels=["Y 1", "Y 2"], xData=x, title="Test", xLabel="Time", yLabel="Value", linewidth=[3, 8])


    def testMultiAxesPlot(self):
        """
        Demonstrate that we can plot two data sets that have been sampled at different rates on a multi-axes plot.
        """
        # Generate the first data set of 2 sine waves.
        data = FunctionGenerator.GetSineWavesAsDataFrame(magnitude=[10, 6, 8, 2], frequency=[4, 10, 2, 1], yOffset=[0, 22, 0, 2], slope=[10, 0, -6, 0], steps=1000)

        # Generate the second data set of 2 sine waves.
        # sine2 = FunctionGenerator.GetSineWavesAsDataFrame(magnitude=[8, 2], frequency=[2, 1], yOffset=[0, 2], slope=[-6, 0], steps=1000)

        data.rename({"y0" : "Sine A1"}, axis="columns", inplace=True)
        data.rename({"y1" : "Sine B1"}, axis="columns", inplace=True)
        data.rename({"y2" : "Sine A2"}, axis="columns", inplace=True)
        data.rename({"y3" : "Sine B2"}, axis="columns", inplace=True)

        figure, axeses = PlotMaker.NewMultiYAxesPlot(data, "x", [["Sine A1", "Sine A2"], ["Sine B1", "Sine B2"]], linewidth="4.0")
        AxesHelper.Label(axeses, title="Multiple Y Axis Plot", xLabel="Time", yLabels=["Left (A)", "Right (B)"])
        figure.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=2, bbox_transform=axeses[0].transAxes)
        plt.show()


if __name__ == "__main__":
    unittest.main()