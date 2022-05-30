"""
Created on May 30, 2022
@author: Lance
"""
import pandas                  as pd
import numpy                   as np

import keras
from   tensorflow.keras.utils  import to_categorical

import os

from lendres.PlotMaker         import PlotMaker

import unittest

class TestTensorFlowHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inputFile  = "insurance.csv"
        inputFile  = os.path.join("Data", inputFile)
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


if __name__ == "__main__":
    unittest.main()