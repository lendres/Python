# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import seaborn as sns

from lendres.PlotHelper import PlotHelper

class BivariateAnalysis:

    @classmethod
    def CreateBivariateHeatMap(cls, data, columns=None, save=False):
        """
        Creates a new figure that has a bar plot labeled with a percentage for a single variable analysis.  Does this
        for every entry in the list of columns.

        Parameters
        ----------
        data : pandas.DataFrame
            The data.
        columns : list of strings
            If specified, only those columns are used for the correlation, otherwise all numeric columns will be used.
        save : bool, optional
            If true, the plots as images.  The default is False.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        axis : matplotlib.pyplot.axis
            The axis of the plot.
        """

        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        # Initialize so the variable is available.
        correlationValues = []

        # If the input argument "columns" is "None," plot all the columns, otherwise, only
        # plot those columns specified in the "columns" argument.
        if columns == None:
            correlationValues = data.corr()
        else:
            correlationValues = data[columns].corr()

        axis = sns.heatmap(correlationValues, annot=True, annot_kws={"fontsize" : 10*PlotHelper.scale}, fmt=".2f")
        axis.set(title="Heat Map for Continuous Data")

        figure = plt.gcf()

        plt.show()

        if save:
            fileName = "Bivariante Heat Map"
            PlotHelper.SavePlot(fileName, figure=figure)

        return figure, axis


    @classmethod
    def CreateBivariatePairPlot(cls, data, columns=None, save=False):
        """
        Creates a new figure that has a bar plot labeled with a percentage for a single variable analysis.  Does this
        for every entry in the list of columns.

        Parameters
        ----------
        data : Pandas DataFrame
            The data.
        columns : List of strings
            If specified, only those columns are used for the correlation, otherwise all numeric columns will be used.
        save : bool, optional
            If true, the plots as images.  The default is False.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        """

        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        if columns == None:
            sns.pairplot(data)
        else:
            sns.pairplot(data[columns])

        figure = plt.gcf()

        figure.suptitle("Pair Plot for Continuous Data", y=1.01)

        plt.show()

        if save:
            fileName = "Bivariante Pair Plot"
            PlotHelper.SavePlot(fileName, figure=figure)

        return figure


    @classmethod
    def PlotComparisonByCategory(cls, data, xColumn, yColumn, sortColumn, title, save=False):
        """
        Creates a scatter plot of a column sorted by another column.

        Parameters
        ----------
        data : Pandas DataFrame
            The data.
        xColumn : string
            Independent variable column in the data.
        yColumn : string
            Dependent variable column in the data.
        sortColumn : string
            Variable column in the data to sort by.
        title : string
            Plot title. The default is 1.0.
        save : bool, optional
            If true, the plots as images.  The default is False.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        axis : matplotlib.pyplot.axis
            The axis of the plot.
        """
        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        axis = sns.scatterplot(x=data[xColumn], y=data[yColumn], hue=data[sortColumn], palette=["indianred","mediumseagreen"])
        axis.set(title=title, xlabel=xColumn.title(), ylabel=yColumn.title())
        axis.get_legend().set_title(sortColumn.title())

        figure = plt.gcf()

        if save:
            fileName = sortColumn + "_" + xColumn + "_verus_" + yColumn + ".png"
            PlotHelper.SavePlot(fileName, figure=figure)

        plt.show()

        return figure, axis