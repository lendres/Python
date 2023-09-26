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


    # @unittest.skip("")
    def testDisplayColor(self):
        """
        Plots all the colors tables with names as labels.
        """
        self.PlotAllTables()


    # @unittest.skip("")
    def testDisplayColorsWithImageSave(self):
        """
        Plots all the colors tables with names as labels and saves them to a file.
        """
        self.PlotAllTables(saveImage=True)


    # @unittest.skip("")
    def testDisplayHexColors(self):
        """
        Plots all the colors tables with hex values as labels and saves them to a file.
        """
        self.PlotAllTables(label="hex")

    
    # @unittest.skip("")
    def testDisplayHexColorsWithImageSave(self):
        """
        Plots all the colors tables with hex values as labels and saves them to a file.
        """
        self.PlotAllTables(label="hex", saveImage=True)

        
    def PlotAllTables(self, **kwargs):
        PlotAllColors("base", **kwargs)
        PlotAllColors("tableau", **kwargs)
        PlotAllColors("css", **kwargs)
        PlotAllColors("xkcd", **kwargs)
        PlotAllColors("full", **kwargs)
        PlotAllColors("seaborn", **kwargs)


if __name__ == "__main__":
    unittest.main()