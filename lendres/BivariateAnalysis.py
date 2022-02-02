# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import seaborn as sns

import lendres

def CreateBiVariateHeatMap(data, columns=None, scale=1.0, save=False):
    """
    Creates a new figure that has a bar plot labeled with a percentage for a single variable analysis.  Does this
    for every entry in the list of categories.

    Parameters
    ----------
    data : Pandas DataFrame
        The data.
    columns : List of strings
        If specified, only those columns are used for the correlation, otherwise all numeric columns will be used.
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

    correlationValues = []
    if columns == None:
        correlationValues = data.corr()
    else:
        correlationValues = data[columns].corr()

    axis = sns.heatmap(correlationValues, annot=True, annot_kws={"fontsize" : 10}, fmt=".2f")
    axis.set(title="Heat Map for Continuous Data")

    figure = plt.gcf()

    plt.show()

    if save:
        fileName = "Bivariante Heat Map"
        lendres.Plotting.SavePlot(fileName, figure=figure, useDefaultOutputFolder=True)

    return figure


def CreateBiVariatePairPlot(data, columns=None, scale=1.0, save=False):
    """
    Creates a new figure that has a bar plot labeled with a percentage for a single variable analysis.  Does this
    for every entry in the list of categories.

    Parameters
    ----------
    data : Pandas DataFrame
        The data.
    columns : List of strings
        If specified, only those columns are used for the correlation, otherwise all numeric columns will be used.
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

    if columns == None:
        sns.pairplot(data)
    else:
        sns.pairplot(data[columns])

    figure = plt.gcf()

    figure.suptitle("Pair Plot for Continuous Data", y=1.01)
    # plt.gcf().subplots_adjust(top=0.95)

    plt.show()

    if save:
        fileName = "Bivariante Pair Plot"
        lendres.Plotting.SavePlot(fileName, figure=figure, useDefaultOutputFolder=True)

    return figure


def PlotComparisonByCategory(data, xCategory, yCategory, sortCategory, title):
    # Must be run before creating figure or plotting data.
    lendres.Plotting.FormatPlot()

    axis = sns.scatterplot(x=data[xCategory], y=data[yCategory], hue=data[sortCategory], palette=['indianred','mediumseagreen'])
    axis.set(title=title, xlabel=xCategory.title(), ylabel=yCategory.title())

    lendres.Plotting.SavePlot(sortCategory + "_" + xCategory + "_verus_" + yCategory + ".png")

    plt.show()