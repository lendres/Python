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
from   lendres.demonstration.FunctionGenerator                       import FunctionGenerator
from   lendres.plotting.AnnotationHelper                             import AnnotationHelper

import unittest


class TestPlotHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        PlotHelper.SetSettings(annotationSize=11)

        cls.x, cls.y = FunctionGenerator.SineWave(magnitude=2, frequency=4, startTime=0, timeLength=4, slope=-0.20, steps=50000)

        cls.axes     = None
        cls.lines    = None
        cls.title    = ""


    def setUp(self):
        self.NewPlot()


    def tearDown(self):
        self.FinishPlot()


    @unittest.skip
    def testMax(self):
        annotationHelper = AnnotationHelper(formatString="{x:0.1f}, {y:0.0f}")
        annotationHelper.AddMaximumAnnotation(self.lines)
        self.title = "Label Max"


    # @unittest.skip
    def testPeaksX(self):
        annotationHelper = AnnotationHelper(formatString="{x:0.1f}")
        annotationHelper.AddPeakAnnotations(self.lines, number=5, sortBy="globalheight")
        self.title = "Label Peaks X"


    @unittest.skip
    def testPeaksY(self):
        annotationHelper = AnnotationHelper(formatString="{y:0.2f}")
        annotationHelper.AddPeakAnnotations(self.lines, number=5, sortBy="globalheight")
        self.title = "Label Peaks Y"


    def testAdjustText1(self):
        annotationHelper = AnnotationHelper(formatString="{x:0.1f}, {y:0.2f}", size="10")
        annotationHelper.SetAdjustText(adjustText=True, arrowprops={"arrowstyle":"-", "color":"red"})
        annotationHelper.AddPeakAnnotations(self.lines, number=5, sortBy="globalheight")
        self.title = "Adjust Text 1"


    def NewPlot(self):
        PlotHelper.Format()

        self.axes   = plt.gca()
        self.lines  = self.axes.plot(self.x, self.y, label="Sine Wave")


    def FinishPlot(self):
        AxesHelper.Label(self.axes, title=self.title, xLabels="Values", yLabels="Count")
        plt.show()



if __name__ == "__main__":
    unittest.main()