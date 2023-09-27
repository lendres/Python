"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import matplotlib.pyplot                                        as plt

#import DataSetLoading
from   lendres.demonstration.FunctionGenerator                  import FunctionGenerator
from   lendres.plotting.PlotHelper                              import PlotHelper
from   lendres.plotting.AxesHelper                              import AxesHelper
from   lendres.plotting.PlotMaker                               import PlotMaker

import unittest


class TestFunctionGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass


    def testSineWaveGenerator(self):
        x, y = FunctionGenerator.GetSineWave()
        self.CreateSineWavePlot(x, y, "No Arguments")

        x, y = FunctionGenerator.GetSineWave(20, 2, 50, 2, 10, 10)
        self.CreateSineWavePlot(x, y, "Use All Arguments")


    def testSineWaveDataFrame(self):
        dataFrame = FunctionGenerator.GetMultipleSineWaveDataFrame()

        # This function is a short cut for creating multiple Y axis plots.  Really this creates multiple "axes" (a set of X and Y axis).
        # Therefore, you have to align all the X values of each axes or it looks funny.  This function createas all the axes and
        # it automatically aligns the X axis of each one.
        # You pass a list of lists as the y values.  Each list are the values to plot on each axes.  This plots "y0" on the first axes,
        # "y1" and "y2" on the second axes, and "y3" on the third.
        figure, axeses = PlotMaker.NewMultiYAxesPlot(dataFrame, "x", [["y0"], ["y1", "y2"], ["y3"]])

        # The AxesHelper can automatically label the axes if you supply it a list of strings for the y labels.
        AxesHelper.Label(axeses, title="Multiple Y Axis Plot", xLabels="Time", yLabels=["Left", "Right 1", "Right 2"])

        figure.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=4, bbox_transform=axeses[0].transAxes)
        plt.show()


    @classmethod
    def CreateSineWavePlot(cls, x, y, title):
        """
        An example plot that is a simple sine wave.

        Parameters
        ----------
        x : array like
            The sine wave x values.
        y : array like
            The sine wave y values.
        title : string
            Title to use for the plot.

        Returns
        -------
        None.
        """
        PlotHelper.Format()

        axes = plt.gca()

        axes.plot(x, y, label="Sine Wave")
        axes.set(title=title, xlabel="Time", ylabel="Vibration")

        plt.show()


if __name__ == "__main__":
    unittest.main()