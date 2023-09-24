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
        PlotMaker.PlotColorCycle(colorStyle="pyplot")
        PlotMaker.PlotColorCycle(colorStyle="seaborn")


    def testMultiAxesPlot(self):
        """
        Demonstrate that we can plot two data sets that have been sampled at different rates on a multi-axes plot.
        """
        # Generate the first data set of 2 sine waves.
        sine1 = FunctionGenerator.GetSineWavesAsDataFrame(magnitude=[10, 6], frequency=[4, 10], yOffset=[0, 22], slope=[10, 0], steps=1000)
        sine1.name = "Data 1"

        # Generate the second data set of 2 sine waves.
        sine2 = FunctionGenerator.GetSineWavesAsDataFrame(magnitude=[8, 2], frequency=[2, 1], yOffset=[0, 2], slope=[-6, 0], steps=1000)
        sine2.name = "Data 2"

        for sine in [sine1, sine2]:
            sine.rename({"y0" : "Sine a"}, axis="columns", inplace=True)
            sine.rename({"y1" : "Sine b"}, axis="columns", inplace=True)

        figure, axeses = PlotMaker.NewMultiYAxesPlot([sine1, sine2], "x", [["Sine a"], ["Sine b"]], linewidth="4.0")
        AxesHelper.Label(axeses, title="Multiple Y Axis Plot", xLabel="Time", yLabels=["Left (a)", "Right (b)"])
        figure.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=2, bbox_transform=axeses[0].transAxes)
        plt.show()


if __name__ == "__main__":
    unittest.main()