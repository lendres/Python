"""
Created on December 27, 2021
@author: Lance A. Endres
"""
import seaborn                                   as sns
import matplotlib.pyplot                         as plt
import numpy                                     as np

from   lendres.plotting.PlotHelper               import PlotHelper
from   lendres.ConsoleHelper                     import ConsoleHelper


class UnivariateAnalysis():

    @classmethod
    def CreateBoxPlot(cls, data, column):
        """
        Creates a bar chart that shows the percentages of each type of entry of a column.

        Parameters
        ----------
        data : pandas.DataFrame
            The data.
        column : string
            Category name in the DataFrame.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        """
        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()
        params = {"figure.figsize" : (PlotHelper.scale*10, PlotHelper.scale*1.25)}
        plt.rcParams.update(params)

        # This creates the bar chart.  At the same time, save the figure so we can return it.
        axis = plt.gca()
        cls.PlotBoxPlot(axis, data, column)

        title = "Column " + "\"" + column + "\""
        axis.set(title=title, xlabel=column, ylabel="Count")

        # Save it so we can return it.  Once "show" is called, the figure is no longer accessible.
        figure = plt.gcf()

        # Make sure the plot is shown.
        plt.show()

        return figure


    @classmethod
    def PlotBoxPlot(cls, axis, data, column, autoLabelX=True):
        """
        Univariate box plot creation.

        Parameters
        ----------
        axis : axis
            Matplotlib axis to plot on.
        data : pandas.DataFrame
            The data.
        column : string
            Category name in the DataFrame.
        autoLabelX : bool
            If true, x axis will be labeled with a name generated from the column.

        Returns
        -------
        None.
        """
        # Boxplot will be created and a star will indicate the mean value of the column.
        sns.boxplot(x=data[column], ax=axis, showmeans=True, color="cyan")

        if autoLabelX:
            axis.set(xlabel=column)
        else:
            axis.set(xlabel=None)


    @classmethod
    def PlotHistogram(cls, axis, data, column, autoLabelX=True, bins=None):
        """
        Univariate histogram creation.

        Parameters
        ----------
        axis : axis
            Matplotlib axis to plot on.
        data : pandas.DataFrame
            The data.
        column : string
            Category name in the DataFrame.
        bins : int
            Size of data bins to use.

        Returns
        -------
        None.
        """
        if bins:
            sns.histplot(data[column], kde=True, ax=axis, bins=bins, palette="winter")
        else:
            sns.histplot(data[column], kde=True, ax=axis, color="grey")

        # Show the mean as vertical line.
        axis.axvline(np.mean(data[column]), color='g', linestyle='--')

        # Label the axis.
        if autoLabelX:
            axis.set(xlabel=column)
        else:
            axis.set(xlabel=None)


    @classmethod
    def CreateBoxAndHistogramPlot(cls, data, column):
        """
        Creates a new figure that has a box plot and histogram for a single variable analysis.

        Parameters
        ----------
        data : pandas.DataFrame
            The data.
        column : string
            Category name in the DataFrame.
        scale : double
            Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        """
        figure, (boxAxis, histogramAxis) = PlotHelper.NewTopAndBottomAxisFigure("Column " + "\"" + column + "\"")

        cls.PlotBoxPlot(boxAxis, data, column, autoLabelX=False)
        cls.PlotHistogram(histogramAxis, data, column)

        plt.show()

        return figure


    @classmethod
    def BoxPlotAndLimitsDisplay(cls, dataHelper, columns, count=5):
        """
        Creates a box plot and displays the min and max values.

        Parameters
        ----------
        dataHelper : DataHelper
            DataHelper that contains the data.
        columns : list of strings
            Columns to generate the display for.
        count : int
            Number of minimum and maximum values to display.

        Returns
        -------
        None.
        """
        for column in columns:
            cls.CreateBoxPlot(dataHelper.data, column)
            minMaxValues = dataHelper.GetMinAndMaxValues(column, count, method="quantity")
            dataHelper.consoleHelper.Display(minMaxValues, ConsoleHelper.VERBOSEREQUESTED)


    @classmethod
    def CreateSideBySideHistogramPlot(cls, leftData, rightData, title, leftTitle, rightTitle, bins=None):
        """
        Creates a box plot and displays the min and max values.

        Parameters
        ----------
        dataHelper : DataHelper
            DataHelper that contains the data.
        columns : list of strings
            Columns to generate the display for.
        count : int
            Number of minimum and maximum values to display.

        Returns
        -------
        None.
        """
        # Create the figure and axes.
        figure, (leftAxis, rightAxis) = PlotHelper.NewSideBySideAxisFigure(title)

        if bins != None:
            sns.histplot(leftData, kde=True, ax=leftAxis, bins=bins, palette="winter")
            sns.histplot(rightData, kde=True, ax=rightAxis, bins=bins,palette="winter")
        else:
            sns.histplot(leftData, kde=True, ax=leftAxis, palette="winter")
            sns.histplot(rightData, kde=True, ax=rightAxis, palette="winter")

        leftAxis.set(xlabel=leftTitle)
        rightAxis.set(xlabel=rightTitle)

        plt.show()

        return figure