# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 20:32:47 2021

@author: Lance
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import lendres.Plotting


def CreateBoxPlot(data, category, scale=1.0):
    """
    Creates a bar chart that shows the percentages of each type of entry of a category.

    Parameters
    ----------
    data : Pandas DataFrame
        The data.
    category: string
        Category name in the DataFrame.
    scale : double
        Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot.

    Returns
    -------
    figure : Figure
        The newly created figure.
    """

    # Must be run before creating figure or plotting data.
    lendres.Plotting.FormatPlot(scale=scale)
    params = {
        "figure.figsize"         : (scale*10, scale*1.25)
    }
    plt.rcParams.update(params)

    # This creates the bar chart.  At the same time, save the figure so we can return it.
    axis = plt.gca()
    PlotBoxPlot(axis, data, category)

    title = "\"" + category.title() + "\"" + " Category"
    axis.set(title=title, xlabel=category.title(), ylabel="Count")

    # Save it so we can return it.  Once "show" is called, the figure is no longer accessible.
    figure = plt.gcf()

    # Make sure the plot is shown.
    plt.show()

    return figure


def PlotBoxPlot(axis, data, category, autoLabelX=True):
    """
    Univariate box plot creation.

    Parameters
    ----------
    axis : axis
        Matplotlib axis to plot on.
    data : Pandas DataFrame
        The data.
    category : string
        Category name in the DataFrame.
    autoLabelX : bool
        If true, x axis will be labeled with a name generated from the category.

    Returns
    -------
    None.
    """

    # Boxplot will be created and a star will indicate the mean value of the column.
    sns.boxplot(x=data[category], ax=axis, showmeans=True, color="cyan")

    if autoLabelX:
        axis.set(xlabel=category.title())
    else:
        axis.set(xlabel=None)


def PlotHistogram(axis, data, category, autoLabelX=True, bins=None):
    """
    Univariate histogram creation.

    Parameters
    ----------
    axis : axis
        Matplotlib axis to plot on.
    data : Pandas DataFrame
        The data.
    category : string
        Category name in the DataFrame.
    bins : int
        Size of data bins to use.

    Returns
    -------
    None.
    """

    if bins:
        sns.histplot(data[category], kde=True, ax=axis, bins=bins, palette="winter")
    else:
        sns.histplot(data[category], kde=True, ax=axis, color="grey")

    # Show the mean as vertical line.
    axis.axvline(np.mean(data[category]), color='g', linestyle='--')

    # Label the axis.
    if autoLabelX:
        axis.set(xlabel=category.title())
    else:
        axis.set(xlabel=None)


def CreateBoxAndHistogramPlot(data, category, scale=1.0):
    """
    Creates a new figure that has a box plot and histogram for a single variable analysis.

    Parameters
    ----------
    data : Pandas DataFrame
        The data.
    category : string
        Category name in the DataFrame.
    scale : double
        Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot.

    Returns
    -------
    figure : Figure
        The newly created figure.
    """

    figure, (boxAxis, histogramAxis) = lendres.Plotting.NewTopAndBottomAxisFigure(category)

    PlotBoxPlot(boxAxis, data, category, autoLabelX=False)
    PlotHistogram(histogramAxis, data, category)

    plt.show()

    return figure


def LabelPercentagesOnCountPlot(axis, data, category, scale=1.0):
    """
    Plot the percentages of each entry of a category.

    Parameters
    ----------
    axis : axis
        Matplotlib axis to plot on.
    data : Pandas DataFrame
        The data.
    category: string
        Category name in the DataFrame.
    scale : double
        Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot.

    Returns
    -------
    None.
    """

    # Number of entries in the category.
    total = len(data[category])

    for patch in axis.patches:
        # Percentage of the category.
        percentage = '{:.1f}%'.format(100*patch.get_height()/total)

        # Find the center of the column/patch on the x-axis.
        x = patch.get_x() + patch.get_width()/2

        # Hieght of the column/patch.  Add a little so it does not touch the top of the column.
        y = patch.get_y() + patch.get_height() + 0.5

        # Plot a label slightly above the column and use the horizontal alignment to center it in the column.
        axis.annotate(percentage, (x, y), size=scale*15, fontweight="bold", horizontalalignment="center")


def CreatePercentageBarPlot(data, category, scale=1.0):
    """
    Creates a bar chart that shows the percentages of each type of entry of a category.

    Parameters
    ----------
    data : Pandas DataFrame
        The data.
    category: string
        Category name in the DataFrame.
    scale : double
        Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot.

    Returns
    -------
    figure : Figure
        The newly created figure.
    """

    # Must be run before creating figure or plotting data.
    lendres.Plotting.FormatPlot(scale=scale)

    # This creates the bar chart.  At the same time, save the figure so we can return it.
    axis   = sns.countplot(x=data[category], palette='winter')
    figure = plt.gcf()

    # Label the individual columns with a percentage, then add the titles to the plot.
    LabelPercentagesOnCountPlot(axis, data, category, scale)

    title = "\"" + category.title() + "\"" + " Category"
    axis.set(title=title, xlabel=category.title(), ylabel="Count")

    # Make sure the plot is shown.
    plt.show()

    return figure