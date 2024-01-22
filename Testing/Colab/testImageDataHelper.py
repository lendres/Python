"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import pandas                                                        as pd
import numpy                                                         as np
import cv2

import os

from   lendres.io.ConsoleHelper                                      import ConsoleHelper
from   lendres.ImageHelper                                           import ImageHelper
from   lendres.ImageDataHelper                                       import ImageDataHelper

import unittest


class TestImageDataHelper(unittest.TestCase):
    #verboseLevel = ConsoleHelper.VERBOSENONE
    #verboseLevel = ConsoleHelper.VERBOSETESTING
    verboseLevel = ConsoleHelper.VERBOSEREQUESTED
    #verboseLevel = ConsoleHelper.VERBOSEIMPORTANT


    @classmethod
    def setUpClass(cls):
        imagesInputFile = "plant-species-images-reduced.npy"
        labelsFile      = "plant-species-labels-reduced.csv"

        imagesInputFile = os.path.join("../Data", imagesInputFile)
        labelsFile      = os.path.join("../Data", labelsFile)

        consoleHelper   = ConsoleHelper(verboseLevel=cls.verboseLevel, useMarkDown=True)
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
        self.imageDataHelper.consoleHelper.PrintNewLine(2, ConsoleHelper.VERBOSEREQUESTED)
        self.imageDataHelper.consoleHelper.PrintTitle("Label Categories", ConsoleHelper.VERBOSEREQUESTED)
        self.imageDataHelper.DisplayLabelCategories()


    def testColorConversion(self):
        self.imageDataHelper.PlotImage(index=0)
        self.imageDataHelper.colorConversion = cv2.COLOR_BGR2RGB
        print("\n")
        self.imageDataHelper.PlotImage(index=0)


    def testDisplayCategoryExamples(self):
        self.imageDataHelper.PlotImageExamplesForAllCategories(numberOfExamples=3)


if __name__ == "__main__":
    unittest.main()