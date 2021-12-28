"""
Created on Sat Dec  4 18:49:50 2021

@author: Lance A. Endres
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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


def SavePlot(saveFileName, useOutputFolder=True):
    """
    Saves a plot with a set of default parameters.

    Parameters
    ----------
    saveFileName : string
        The (optionally) path and file name to save the image to.
    useOutputFolder : bool
        If true, the image is saved to a subfolder or the current folder called "Output."

    Returns
    -------
    None.
    """
    path = ".\\Output\\" + saveFileName
    plt.savefig(path, dpi=500, transparent=True, bbox_inches="tight")