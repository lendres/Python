# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 20:32:47 2021

@author: Lance
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import lendres.Plotting


def CreateUnivariateBoxPlot(axis, data, category):
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

    Returns
    -------
    None.
    """

    # Boxplot will be created and a star will indicate the mean value of the column.
    sns.boxplot(x=data[category], ax=axis, showmeans=True, color="cyan")
    axis.set(xlabel=None)


def CreateUnivariateHistogram(axis, data, category, bins=None):
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
        sns.histplot(data[category], kde=False, ax=axis, bins=bins, palette="winter")
    else:
        sns.histplot(data[category], kde=False, ax=axis, color="grey")

    # Show the mean as vertical line.
    axis.axvline(np.mean(data[category]), color='g', linestyle='--')

    # Label the axis.
    axis.set(xlabel=category.title())


def UnivariateBoxAndHistogramPlot(data, category, scale=1.0):
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

    CreateUnivariateBoxPlot(boxAxis, data, category)
    CreateUnivariateHistogram(histogramAxis, data, category)

    plt.show()

    return figure


def MakeBoxAndHistogramPlots(data, categories, save=False):
    """
    Creates a new figure that has a box plot and histogram for a single variable analysis.  Does this
    for every entry in the list of categories.

    This is the main entry point for creating the box and histogram plots.

    Parameters
    ----------
    data : Pandas DataFrame
        The data.
    category : an arry or list of strings
        Category names in the DataFrame.
    save : bool
        If true, the plots are saved to the default plotting directory.

    Returns
    -------
    None.
    """

    for category in categories:
        figure = UnivariateBoxAndHistogramPlot(data, category)

        if save:
            fileName = "Box and Histogram Plot " + category.title() + " Category"
            lendres.Plotting.SavePlot(fileName, figure=figure, useDefaultOutputFolder=True)


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


def MakePercentageBarPlots(data, categories, save=False):
    """
    Creates a new figure that has a bar plot labeled with a percentage for a single variable analysis.  Does this
    for every entry in the list of categories.

    Parameters
    ----------
    data : Pandas DataFrame
        The data.
    category : an arry or list of strings
        Category names in the DataFrame.
    save : bool
        If true, the plots are saved to the default plotting directory.

    Returns
    -------
    None.
    """
    for category in categories:
        figure = CreatePercentageBarPlot(data, category)

        if save:
            fileName  = "Percentage Bar Plot " + category.title() + " Category"
            lendres.Plotting.SavePlot(fileName, figure=figure, useDefaultOutputFolder=True)