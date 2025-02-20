"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import numpy                                                    as np
import matplotlib.pyplot                                        as plt

from   lendres.plotting.AxesHelper                              import AxesHelper
from   lendres.plotting.PlotHelper                              import PlotHelper
from   lendres.plotting.PlotMaker                               import PlotMaker
from   lendres.demonstration.FunctionGenerator                  import FunctionGenerator

import unittest


class TestPlotMaker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.confusionMatrix3x3 = np.array(
            [[25,  5,  10],
             [ 6, 10,   4],
             [ 8,  6,  15]]
        )

        # Generate a data set of 4 sine waves.
        cls.sinesDataFrame = FunctionGenerator.SineWavesAsDataFrame(magnitude=[10, 6, 8, 2], frequency=[4, 8, 2, 1], yOffset=[0, 22, 0, 2], slope=[10, 0, -6, 0], steps=1000)

        cls.sinesDataFrame.rename({"y0" : "Sine A1"}, axis="columns", inplace=True)
        cls.sinesDataFrame.rename({"y1" : "Sine B1"}, axis="columns", inplace=True)
        cls.sinesDataFrame.rename({"y2" : "Sine A2"}, axis="columns", inplace=True)
        cls.sinesDataFrame.rename({"y3" : "Sine B2"}, axis="columns", inplace=True)


    # @unittest.skip
    def testConfusionMatrix(self):
        PlotMaker.CreateConfusionMatrixPlot(self.confusionMatrix3x3, "3 by 3 Confusion Matrix")
        PlotMaker.colorMap = "Blues"
        PlotMaker.CreateConfusionMatrixPlot(self.confusionMatrix3x3, "3 by 3 Confusion Matrix")
        PlotMaker.colorMap = None
        labels = ["car", "boat", "train"]
        PlotMaker.CreateConfusionMatrixPlot(self.confusionMatrix3x3, "3 by 3 Confusion Matrix", axesLabels=labels)


    # @unittest.skip
    def testPlotColorCycle(self):
        # A test that will also conveniently display the color cycles for reference.
        PlotMaker.CreateColorCyclePlot(lineColorCycle="pyplot")
        PlotMaker.CreateColorCyclePlot(lineColorCycle="seaborn")


    # @unittest.skip
    def testCreateFastFigure(self):
        x, a = FunctionGenerator.SineWave(magnitude=10, frequency=4, yOffset=0, slope=0, steps=1000)
        x, b = FunctionGenerator.SineWave(magnitude=4, frequency=2, yOffset=0, slope=10, steps=1000)
        x, c = FunctionGenerator.SineWave(magnitude=5, frequency=3, yOffset=30, slope=-5, steps=1000)
        PlotMaker.CreateFastFigure([a])
        PlotMaker.CreateFastFigure([a, b], yDataLabels=["Y 1", "Y 2"], xData=x, title="Test Fast Figure 1 Kwarg", xLabel="Time", yLabel="Value", linewidth=7)
        PlotMaker.CreateFastFigure([a, b], yDataLabels=["Y 1", "Y 2"], xData=x, title="Test Fast Figure List Kwarg", xLabel="Time", yLabel="Value", linewidth=[3, 8])


    # @unittest.skip
    def testCreateSimpleFastFigure(self):
        x, a = FunctionGenerator.SineWave(magnitude=10, frequency=4, yOffset=0, slope=0, steps=1000)
        x, b = FunctionGenerator.SineWave(magnitude=4, frequency=2, yOffset=0, slope=10, steps=1000)
        x, c = FunctionGenerator.SineWave(magnitude=5, frequency=3, yOffset=30, slope=-5, steps=1000)
        PlotMaker.CreateSimpleFastFigure(a, title="Test Simple Fast Figure")
        PlotMaker.CreateSimpleFastFigure(a, yDataLabel="Y 1", xData=x, title="Test Simple Fast Figure Kwarg", xLabel="Time", yLabel="Value", linewidth=7)


    def testMultiAxesPlot(self):
        """
        Demonstrate multi-axeses plotting.
        """
        self.PlotMultiAxes("No Key Word Arguments")


    # @unittest.skip
    def testKeywordArgsForMultiAxesPlot(self):
        self.PlotMultiAxes("Key Word Argument as Int", linewidth=5.0)
        self.PlotMultiAxes("Key Word Argument as List", linewidth=[5.5, 3.5, 1.5, 7.5])
        self.PlotMultiAxes("Mixed Key Word Arguments", linewidth=[5.5, 3.5, 1.5, 7.5], linestyle="dashed")


    # @unittest.skip
    def testBracketsOnColumnNames(self):
        self.PlotMultiAxes("Mixed Brackets", columns=["Sine A1", ["Sine B1", "Sine B2"]])
        self.PlotMultiAxes("No Brackets", columns=["Sine A1", "Sine B1"])


    def PlotMultiAxes(self, titleSuffix, columns=[["Sine A1", "Sine A2"], ["Sine B1", "Sine B2"]], **kwargs):
        PlotHelper.Format()
        figure, axeses = PlotMaker.NewMultiYAxesPlot(self.sinesDataFrame, "x", columns, **kwargs)
        AxesHelper.Label(axeses, title="Multiple Y Axis Plot\n"+titleSuffix, xLabels="Time", yLabels=["Left (A)", "Right (B)"])
        figure.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=2, bbox_transform=axeses[0].transAxes)
        plt.show()


if __name__ == "__main__":
    unittest.main()