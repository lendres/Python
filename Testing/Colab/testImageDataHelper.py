"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import pandas                                    as pd
import numpy                                     as np

import os

from   lendres.ConsoleHelper                     import ConsoleHelper
from   lendres.ImageHelper                       import ImageHelper
from   lendres.ImageDataHelper                   import ImageDataHelper

import unittest

class TestImageDataHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        imagesInputFile = "plant-species-images-reduced.npy"
        labelsFile      = "plant-species-labels-reduced.csv"

        imagesInputFile = os.path.join("../Data", imagesInputFile)
        labelsFile      = os.path.join("../Data", labelsFile)

        consoleHelper   = ConsoleHelper(verboseLevel=ConsoleHelper.VERBOSEALL, useMarkDown=False)
        cls.imageHelper = ImageDataHelper(consoleHelper=consoleHelper)
        cls.imageHelper.LoadImagesFromNumpyArray(imagesInputFile);
        cls.imageHelper.LoadLabelsFromCsv(labelsFile);


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.imageDataHelper = TestImageDataHelper.imageHelper.Copy()


    def testDisplayData(self):
        self.imageDataHelper.DisplayDataShapes()


if __name__ == "__main__":
    unittest.main()