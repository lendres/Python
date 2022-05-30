"""
Created on December 27, 2021
@author: Lance
"""
import pandas                  as pd
import numpy                   as np

import os

from lendres.PlotMaker         import PlotMaker

import unittest

class TestPlotMaker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inputFile = "insurance.csv"
        inputFile = os.path.join("Data", inputFile)
        cls.data   = pd.read_csv(inputFile)

        cls.confusionMatrix3x3 = np.array(
            [[25,  5,  10],
             [ 6, 10,   4],
             [ 8,  6,  15]]
        )


    def testConfusionMatrix(self):
        PlotMaker.CreateConfusionMatrixPlot(TestPlotMaker.confusionMatrix3x3, "3 by 3 Confusion Matrix")


if __name__ == "__main__":
    unittest.main()