"""
Created on Sat March 18, 2023
@author: Lance A. Endres
Based on: https://petercbsmith.github.io/color-tutorial.html
"""
import os

import numpy                                                              as np
import matplotlib                                                         as mpl
import matplotlib.pyplot                                                  as plt
import matplotlib.colors                                                  as mcolors
from   matplotlib.colors                                                  import LinearSegmentedColormap
from   matplotlib.colors                                                  import ListedColormap
from   matplotlib.gridspec                                                import GridSpec
import seaborn                                                            as sns

from   lendres.plotting.PlotHelper                                        import PlotHelper

class ColorHelper():

    @classmethod
    def PlotAllColors(cls, colorTable, label="names", saveImage=False):
        """
        Creates an image of all the colors.

        Parameters
        ----------
        colors : string
            Color table to plot.
                "base"    : mcolors.BASE_COLORS
                "tableau" : mcolors.TABLEAU_COLORS
                "css"     : mcolors.CSS4_COLORS
                "xkcd"    : mcolors.XKCD_COLORS
                "full"    : mcolors._colors_full_map
                "seaborn" : sns.color_palette() (Seaborn default palette)
        saveImage : bool, optional
            If True, the image is saved to the disk. The default is False.

        Returns
        -------
        None.
        """
        # Constants.
        COLUMNWIDTH     = 17.0 / 6
        ROWHEIGHT       = 46.0 / 195

        # Reset Matplotlib in case a non-standard formatting specification is in use or Seaborn colors have been specified, et cetera.
        plt.rcParams.update(plt.rcParamsDefault)

        sort = True

        match colorTable:
            case "base":
                # These colors can be called with a single character.
                colors = mcolors.BASE_COLORS
            case "tableau":
                # The default color cycle colors.
                colors = mcolors.TABLEAU_COLORS
            case "css":
                # Named colors also recognized in CSS.
                colors = mcolors.CSS4_COLORS
            case "xkcd":
                # Named colors from the xkcd survey.
                colors = mcolors.XKCD_COLORS
            case "full":
                # Dictionary of all colors.
                colors = colors = mcolors._colors_full_map
            case "seaborn":
                sns.set(color_codes=True)
                colors = sns.color_palette(as_cmap=True)
                colors = {"color"+str(i) : colors[i] for i in range(len(colors))}
                sort   = False
            case _:
                raise Exception("Unknown color set.")

        # print("\n\ncolors", colors)

        # HSV colors.  Switch to HSV for when we are sorting.
        hsvColors = [
            (
                tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                name if label=="names" else PlotHelper.RgbToHex(mcolors.to_rgb(color))
            )
            for name, color in colors.items()
        ]

        if sort:
            hsvColors = sorted(hsvColors)

        sortedNames     = [name for hsv, name in hsvColors]
        numberOfColors  = len(sortedNames)

        # Establish 6 as our number of columns, then calculate the number of rows.  We add one to the number of colors
        # to allow for adding the "legend" if required.
        numberOfColumns = 6
        numberOfRows    = int(np.ceil((numberOfColors+1) / numberOfColumns))

        figure  = plt.figure(figsize=(numberOfColumns*COLUMNWIDTH, numberOfRows*ROWHEIGHT), dpi=200)
        grid    = GridSpec(numberOfRows, numberOfColumns, hspace=1, wspace=2)

        counter = 0
        hasXkcd = False

        while counter < numberOfColors:
            column = (counter // numberOfRows) % numberOfColumns
            row    = counter % numberOfRows
            axis   = figure.add_subplot(grid[row, column])

            axis.axis("off")
            #(hsvColors[counter])[0]
            #mcolors.to_rgba
            #sortedNames[counter]
            axis.axhline(0, linewidth=30, c=mcolors.hsv_to_rgb((hsvColors[counter])[0]))
            colorName = sortedNames[counter]

            # Recall that XKCD colors must be prefaced with "xkdc:".  To save space, I'll take that out
            # and replace with an asterisk so we can still identify them.
            if "xkcd" in colorName:
                colorName = colorName[5:] + "*"
                hasXkcd   = True

            axis.text(-0.03, 0.5, colorName, ha="right", va="center")
            counter += 1

        if hasXkcd:
            # Place it in the last row and last column.
            axis = figure.add_subplot(grid[-1, -1])
            axis.axis("off")
            axis.text(0, 0, "* = xkcd")

        if saveImage:
            directory = os.path.dirname(os.path.abspath(__file__))
            file      = "Matplotlib Colors - " + colorTable.upper() + " - " + label.title() + ".png"
            path      = os.path.join(directory, file)
            figure.savefig(path, bbox_inches="tight")

        plt.show()


    @classmethod
    def PlotExampleOfColorMap(cls, colorMaps):
        """
        Helper function to plot data with associated colormap.

        From:
            https://matplotlib.org/stable/users/explain/colors/colormap-manipulation.html#colormap-manipulation
        """
        if not type(colorMaps) is list:
            colorMaps = [colorMaps]

        np.random.seed(19680801)
        data                = np.random.randn(30, 30)
        numberOfColorMaps   = len(colorMaps)

        figure, axeses = plt.subplots(1, numberOfColorMaps, figsize=(numberOfColorMaps*2+2, 3), layout="constrained", squeeze=False)
        for [axes, colorMap] in zip(axeses.flat, colorMaps):
            psm = axes.pcolormesh(data, cmap=colorMap, rasterized=True, vmin=-4, vmax=4)
            figure.colorbar(psm, ax=axes)
            axes.set(title=colorMap.name)
        plt.show()
        return figure


    @classmethod
    def GetColorMap(cls, colorMap):
        match colorMap.lower():
            case "greenorange":
                oranges = mpl.colormaps['Oranges'].resampled(128)
                greens  = mpl.colormaps['Greens_r'].resampled(128)

                newcolors = np.vstack((greens(np.linspace(0, 1, 128)), oranges(np.linspace(0, 1, 128))))
                return ListedColormap(newcolors, name="GreenOrange")

            case "greenred1":
                colors = [
                    [0.15294118, 0.39215686, 0.09803922, 1.00],
                    [0.78805544, 0.84536235, 0.93890483, 1.00],
                    [0.92993929, 0.81982561, 0.76085924, 1.00],
                    [0.70567316, 0.01555616, 0.15023281, 1.00]
                ]
                return LinearSegmentedColormap.from_list("GreenRed1", colors)

            case "greenred2":
                colors = [
                    [0.00,       0.50,       0.40,       1.00],
                    [0.78805544, 0.84536235, 0.93890483, 1.00],
                    [0.92993929, 0.81982561, 0.76085924, 1.00],
                    [0.70567316, 0.01555616, 0.15023281, 1.00]
                ]
                return LinearSegmentedColormap.from_list("GreenRed2", colors)

            case "greenred3":
                colors = [
                    [0.00,       0.50,       0.40,       1.00],
                    [0.42857143, 0.71428571, 0.40,       1.00],
                    [0.96726017, 0.6566848,  0.53727972, 1.00],
                    [0.70567316, 0.01555616, 0.15023281, 1.00]
                ]
                return LinearSegmentedColormap.from_list("GreenRed3", colors)

            case "greenred4":
                colors = [
                    [0.00,       0.50,       0.40,       1.00],
                    [0.70,       0.70,       0.70,       1.00],
                    [0.70567316, 0.01555616, 0.15023281, 1.00]
                ]
                return LinearSegmentedColormap.from_list("GreenRed4", colors)

            case "greenyellowred":
                colors = [
                    [0.00,       0.50,       0.40,       1.00],
                    [0.70,       0.70,       0.50,       1.00],
                    [0.70567316, 0.01555616, 0.15023281, 1.00]
                ]
                return LinearSegmentedColormap.from_list("GreenYellowRed", colors)

            case _:
                return mpl.colormaps[colorMap]