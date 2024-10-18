"""
Created on July 23, 2023
@author: Lance A. Endres
"""
import numpy                                                              as np
import matplotlib                                                         as mpl

from   matplotlib.colors                                                  import LinearSegmentedColormap
from   matplotlib.colors                                                  import ListedColormap

from   lendres.io.ConsoleHelper                                           import ConsoleHelper
from   lendres.plotting.ColorHelper                                       import ColorHelper

import unittest

# More information at:
# https://docs.python.org/3/library/unittest.html
skipTests  = 0
saveImages = False

class TestColorHleper(unittest.TestCase):


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
    @unittest.skipIf(not saveImages, "Skip image saving")
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
    @unittest.skipIf(not saveImages, "Skip image saving")
    def testDisplayHexColorsWithImageSave(self):
        """
        Plots all the colors tables with hex values as labels and saves them to a file.
        """
        self.PlotAllTables(label="hex", saveImage=True)


    def testColorMapPlot(self):
        # https://matplotlib.org/stable/gallery/color/colormap_reference.html
        # https://matplotlib.org/stable/users/explain/colors/colormap-manipulation.html#colormap-manipulation

        # coolwarm = mpl.colormaps["coolwarm"].resampled(8)
        # summer   = mpl.colormaps["summer"].resampled(8)

        # print("\ncoolwarm(range(8))\n", coolwarm(range(8)))
        # print("\nsummer(range(8))\n", summer(range(8)))

        # colors = ["darkorange", "gold", "lawngreen", "lightseagreen"]
        # cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
        # ColorHelper.PlotExampleOfColorMap(cmap1)

        ColorHelper.PlotExampleOfColorMap([ColorHelper.GetColorMap("coolwarm"), ColorHelper.GetColorMap("greenorange")])

        ColorHelper.PlotExampleOfColorMap([
            ColorHelper.GetColorMap("coolwarm"),
            ColorHelper.GetColorMap("GreenRed1"),
            ColorHelper.GetColorMap("greenred2"),
            ColorHelper.GetColorMap("greenred3"),
            ColorHelper.GetColorMap("greenred4")
        ])

        ColorHelper.PlotExampleOfColorMap([
            ColorHelper.GetColorMap("greenyellowred"),
            ColorHelper.GetColorMap("greenred4")
        ])

    def PlotAllTables(self, **kwargs):
        ColorHelper.PlotAllColors("base", **kwargs)
        ColorHelper.PlotAllColors("tableau", **kwargs)
        ColorHelper.PlotAllColors("css", **kwargs)
        ColorHelper.PlotAllColors("xkcd", **kwargs)
        ColorHelper.PlotAllColors("full", **kwargs)
        ColorHelper.PlotAllColors("seaborn", **kwargs)


if __name__ == "__main__":
    unittest.main()