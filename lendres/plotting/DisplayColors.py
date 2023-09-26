"""
Created on Sat March 18, 2023
@author: Lance A. Endres
Based on: https://petercbsmith.github.io/color-tutorial.html
"""
import numpy                                                              as np
import matplotlib.pyplot                                                  as plt
import matplotlib.colors                                                  as mcolors
from   matplotlib.gridspec                                                import GridSpec
import seaborn                                                            as sns

from   lendres.plotting.PlotHelper                                        import PlotHelper

# Constants.
COLUMNWIDTH     = 17.0 / 6
ROWHEIGHT       = 46.0 / 195

def PlotAllColors(colorTable, label="name", saveImage=False):
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
            name if label=="name" else PlotHelper.RgbToHex(mcolors.to_rgb(color))
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
        figure.savefig("Matplotlib Named Colors - "+colorTable+".png", bbox_inches="tight")

    plt.show()