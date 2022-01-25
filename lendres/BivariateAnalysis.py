# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import seaborn as sns

import lendres

def CreateBiVariateHeatMap(data, scale=1.0, save=False):
    """
    Creates a new figure that has a bar plot labeled with a percentage for a single variable analysis.  Does this
    for every entry in the list of categories.

    Parameters
    ----------
    data : Pandas DataFrame
        The data.
    scale : double
        Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot.
    save : bool
        If true, the plots are saved to the default plotting directory.

    Returns
    -------
    figure : Figure
        The newly created figure.
    """

     # Must be run before creating figure or plotting data.
    lendres.Plotting.FormatPlot(scale=scale)

    axis = sns.heatmap(data.corr(), annot=True, annot_kws={"fontsize" : 10}, fmt=".2f")
    axis.set(title="Heat Map for Continuous Data")

    figure = plt.gcf()

    plt.show()

    if save:
        fileName = "Bivariante Heat Map"
        lendres.Plotting.SavePlot(fileName, figure=figure, useDefaultOutputFolder=True)

    return figure


def CreateBiVariatePairPlot(data, scale=1.0, save=False):
    """
    Creates a new figure that has a bar plot labeled with a percentage for a single variable analysis.  Does this
    for every entry in the list of categories.

    Parameters
    ----------
    data : Pandas DataFrame
        The data.
    scale : double
        Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot.
    save : bool
        If true, the plots are saved to the default plotting directory.

    Returns
    -------
    figure : Figure
        The newly created figure.
    """

    # Must be run before creating figure or plotting data.
    lendres.Plotting.FormatPlot(scale=scale)

    sns.pairplot(data)

    figure = plt.gcf()

    figure.suptitle("Pair Plot for Continuous Data", y=1.01)
    # plt.gcf().subplots_adjust(top=0.95)

    figure = plt.gcf()

    plt.show()

    if save:
        fileName = "Bivariante Pair Plot"
        lendres.Plotting.SavePlot(fileName, figure=figure, useDefaultOutputFolder=True)


    return figure