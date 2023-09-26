"""
Created on July 23, 2023
@author: Lance A. Endres
"""
from   lendres.ConsoleHelper                                              import ConsoleHelper
from   lendres.plotting.DisplayColors                                     import PlotAllColors

import unittest

# More information at:
# https://docs.python.org/3/library/unittest.html

class TestBoundingDataType(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        pass


    @unittest.skip("")
    def testDisplayColor(self):
        PlotAllColors("base")
        PlotAllColors("tableau")
        PlotAllColors("css")
        PlotAllColors("xkcd")
        PlotAllColors("full")
        PlotAllColors("seaborn")


    @unittest.skip("")
    def testDisplayColorsWithImageSave(self):
        saveImage = True
        PlotAllColors("base", saveImage=saveImage)
        PlotAllColors("tableau", saveImage=saveImage)
        PlotAllColors("css", saveImage=saveImage)
        PlotAllColors("xkcd", saveImage=saveImage)
        PlotAllColors("full", saveImage=saveImage)
        PlotAllColors("seaborn", saveImage=saveImage)

    
    # @unittest.skip("")
    def testDisplayHexColorsWithImageSave(self):
        saveImage = True
        PlotAllColors("base", label="hex", saveImage=saveImage)
        PlotAllColors("tableau", label="hex", saveImage=saveImage)
        PlotAllColors("css", label="hex", saveImage=saveImage)
        PlotAllColors("xkcd", label="hex", saveImage=saveImage)
        PlotAllColors("full", label="hex", saveImage=saveImage)
        PlotAllColors("seaborn", label="hex", saveImage=saveImage)


if __name__ == "__main__":
    unittest.main()