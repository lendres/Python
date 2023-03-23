"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import pandas                                    as pd
import numpy                                     as np

import os

import DataSetLoading
from   lendres.plotting.PlotMaker                import PlotMaker

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
        PlotMaker.CreateConfusionMatrixPlot(TestPlotMaker.confusionMatrix3x3, "3 by 3 Confusion Matrix")
        PlotMaker.colorMap = "Blues"
        PlotMaker.CreateConfusionMatrixPlot(TestPlotMaker.confusionMatrix3x3, "3 by 3 Confusion Matrix")
        PlotMaker.colorMap = None
        labels = ["car", "boat", "train"]
        PlotMaker.CreateConfusionMatrixPlot(TestPlotMaker.confusionMatrix3x3, "3 by 3 Confusion Matrix", axesLabels=labels)

        print(list(range(5)))


if __name__ == "__main__":
    unittest.main()