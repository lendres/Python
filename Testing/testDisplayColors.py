"""
Created on July 23, 2023
@author: Lance A. Endres
"""
from   lendres.io.ConsoleHelper                                      import ConsoleHelper
from   lendres.plotting.DisplayColors                                import PlotAllColors

import unittest

# More information at:
# https://docs.python.org/3/library/unittest.html
skipTests = 1

class TestBoundingDataType(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        pass


    @unittest.skipIf(skipTests, "Time saving")
    def testDisplayColor(self):
        """
        Plots all the colors tables with names as labels.
        """
        self.PlotAllTables()


    @unittest.skipIf(skipTests, "Time saving")
    def testDisplayColorsWithImageSave(self):
        """
        Plots all the colors tables with names as labels and saves them to a file.
        """
        self.PlotAllTables(saveImage=True)


    @unittest.skipIf(skipTests, "Time saving")
    def testDisplayHexColors(self):
        """
        Plots all the colors tables with hex values as labels and saves them to a file.
        """
        self.PlotAllTables(label="hex")


    @unittest.skipIf(skipTests, "Time saving")
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