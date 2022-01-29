"""
Created on Sat Dec  4 18:49:50 2021

@author: Lance A. Endres
"""

import matplotlib.pyplot as plt
import os
import shutil

# def LabelPlot(plot, title, xAxis, yAxis):
#     print(type(plot))
# #    if isinstance(plot, matplotlib.axes._subplots.AxesSubplot):
# #        print("\nTrue")
#     if type(plot) == "<class 'matplotlib.axes._subplots.AxesSubplot'>":
#         print("\nPlot is an axis.")
#         LabelAxis(plot, title, xAxis, yAxis)
#     elif type(plot) == "<class 'matplotlib.figure.Figure'>":
#         print("\nPlot is a figure.")
#         LabelPlot(plot, title, xAxis, yAxis)

# palette=['indianred','mediumseagreen']

def FormatPlot(scale=1.0, transparentLegend=False):
    """
    Sets the font sizes, weights, and other properties of a plot.

    Parameters
    ----------
    scale : double
        Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot.
    transparentLegend : bool
        Option to create a legend with a transparent background.

    Returns
    -------
    None.
    """

    # Standard size.
    size = 20

    # Standard formating parameters.
    params = {
        "figure.figsize"         : (scale*10, scale*6),
        "font.size"              : 1.2*scale*size,
        "font.weight"            : "bold",
        "figure.titlesize"       : 1.2*scale*size,
        "figure.titleweight"     : "bold",
        "legend.fontsize"        : 0.8*scale*size,
        "legend.title_fontsize"  : 0.85*scale*size,
        "legend.edgecolor"       : "black",
        "axes.titlesize"         : 1.1*scale*size,
        "axes.titleweight"       : "bold",
        "axes.labelweight"       : "bold",
        "axes.labelsize"         : scale*size,
        "xtick.labelsize"        : 0.80*scale*size,
        "ytick.labelsize"        : 0.80*scale*size,
        "axes.titlepad"          : 25
    }

    # Parameters to create a legend with a border and transparent background.
    transparentLegendParams = {
        "legend.framealpha"      : None,
        "legend.facecolor"       : (1, 1, 1, 0)
    }

    # Parameters for a legend with a border and a solid background.
    nonTransparentLegendParams = {
        "legend.framealpha"      : 1.0,
        "legend.facecolor"       : 'inherit'
    }

    if transparentLegend:
        params.update(transparentLegendParams)
    else:
        params.update(nonTransparentLegendParams)

    plt.rcParams.update(params)


def NewTopAndBottomAxisFigure(category, topPercent=0.25, scale=1.0):
    """
    Creates a new figure that has two axes, one above another.

    Parameters
    ----------
    category : string
        Category name in the DataFrame.
    scale : double
        Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot.

    Returns
    -------
    figure : Figure
        The newly created figure.
    (boxAxis, historgramAxis) : axis array
        The top axis and bottom axis, respectively, for the box plot and histogram.
    """

    # Check input.
    if topPercent <= 0 or topPercent >= 1.0:
        raise Exception("Top percentage out of rage.")

    # The format setup needs to be run first.
    FormatPlot(scale=scale)

    figure, (boxAxis, histogramAxis) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (topPercent, 1-topPercent)})

    figure.suptitle("\"" + category.title() + "\"" + " Category")

    return (figure, (boxAxis, histogramAxis))



# Default output directory for saving plots.
defaultOutputDirector = ".\\Output\\"


def DeleteOutputDirectory():
    """
    Removes all the files and subdirectories in the default output directory and then deletes the directory.

    Parameters
    ----------
    None.

    Returns
    -------
    None.
    """
    if os.path.isdir(defaultOutputDirector):
        shutil.rmtree(defaultOutputDirector)


def SavePlot(saveFileName, figure=None, useDefaultOutputFolder=False):
    """
    Saves a plot with a set of default parameters.

    Parameters
    ----------
    saveFileName : string
        The (optionally) path and file name to save the image to.
    figure : Figure
        The figure to save.  If "None" is specified, the current figure will be used.
    useDefaultOutputFolder : bool
        If true, the image is saved to a subfolder or the current folder called "Output."

    Returns
    -------
    None.
    """

    if figure == None:
        figure=plt.gcf()

    # Default is to use the save path and file name exactly as it was passed.
    path = saveFileName

    # If the default ouptput folder is specified, we need to make sure it exists and update
    # the save path to account for it.
    if useDefaultOutputFolder:

        # Directory needs to exist.
        if not os.path.isdir(defaultOutputDirector):
            os.mkdir(defaultOutputDirector)

        # Update path.
        path = defaultOutputDirector + saveFileName

    # And, finally, get down to the work.
    figure.savefig(path, dpi=500, transparent=True, bbox_inches="tight")