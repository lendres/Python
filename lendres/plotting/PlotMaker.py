"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import pandas                                                        as pd
import numpy                                                         as np
import matplotlib.pyplot                                             as plt

import seaborn                                                       as sns
sns.set(color_codes=True)

from   lendres.plotting.AxesHelper                                   import AxesHelper
from   lendres.plotting.PlotHelper                                   import PlotHelper
from   lendres.datatypes.ListTools                                   import ListTools

class PlotMaker():
    """
    A class for making common plots.

    Nomenclature:
        Format
            Establishes the font size, line widths, marker sizes, and colors.
        New*Figure
            Makes a new figure, formats the plot, and does additional setup work.
        New*Plot
            Makes a new figure, formats it, does additional setup, and plots data.  Does NOT finalize the figure.
        Finalize
            Shows the plot, the figure can no longer be manipulated after this.  It can be saved as an image.
        Create*
            Makes the entire figure.  The figure is made and formatted, the data plotted, and the figure is finalized.

    """
    # Class level variables.
    # Color map to use for plots.
    colorMap      = None


    @classmethod
    def CreateSimpleFastFigure(cls, yData:list, yDataLabel:str=None, xData=None, title=None, xLabel=None, yLabel=None, showLegend=True, show=True, **kwargs):
        """
        Easly create a basic plot.  While intended to make simple plots fast and easy, a number of options are available
        to customize the plot.

        Parameters
        ----------
        yData : array like
            A list of data sets to plot.
        yDataLabels : string, optional
            Labels to use in the legend for each series. The default is None.
        xData : array like, optional
            The x-axis values.  If none is supplied, a list of integers of the length of the y-data
            is created. The default is None.
        title : string, optional
            Top title for the figure. The default is None.
        xLabel : string, optional
            X-axis title/label. The default is None.
        yLabel : string, optional
            Y-axis title/label. The default is None.
        showLegend : boolean, optional
            Specifies if the legend should be shown. The default is True.
        show : boolean, optional
            If true, the plot is shown. The default is True.
        **kwargs : key word arguments with array like values
            Arguments to pass to each series when it is plotted.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        axeses : tuple of matplotlib.axes.Axes
            The axes of the plot.
        """
        # Must be run before creating figure or plotting data.
        PlotHelper.Format()

        figure = plt.gcf()
        axes   = plt.gca()

        # Handle optional xData.  If none exist, create a set of integers from 1...N where N is the length of the y data.
        if xData is None:
            xData = range(1, len(yData[0])+1)

        # Need to repackage all the key word arguments.
        axes.plot(xData, yData, label=yDataLabel, **kwargs)

        # Label the plot.
        AxesHelper.Label(axes, title=title, xLabels=xLabel, yLabels=yLabel)

        if showLegend:
            figure.legend(loc="upper left", bbox_to_anchor=(0, -0.12*PlotHelper.formatSettings.Scale), ncol=2, bbox_transform=axes.transAxes)

        if show:
            plt.show()

        return figure, axes


    @classmethod
    def CreateFastFigure(cls, yData:list, yDataLabels:list=None, xData=None, title=None, xLabel=None, yLabel=None, showLegend=True, show=True, **kwargs):
        """
        Easly create a basic plot.  While intended to make simple plots fast and easy, a number of options are available
        to customize the plot.

        Parameters
        ----------
        yData : array like of array like
            A list of data sets to plot.
        yDataLabels : array like of strings, optional
            Labels to use in the legend for each series. The default is None.
        xData : array like, optional
            The x-axis values.  If none is supplied, a list of integers of the length of the y-data
            is created. The default is None.
        title : string, optional
            Top title for the figure. The default is None.
        xLabel : string, optional
            X-axis title/label. The default is None.
        yLabel : string, optional
            Y-axis title/label. The default is None.
        showLegend : boolean, optional
            Specifies if the legend should be shown. The default is True.
        show : boolean, optional
            If true, the plot is shown. The default is True.
        **kwargs : key word arguments with array like values
            Arguments to pass to each series when it is plotted.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        axeses : tuple of matplotlib.axes.Axes
            The axes of the plot.
        """
        # Must be run before creating figure or plotting data.
        PlotHelper.Format()

        figure = plt.gcf()
        axes   = plt.gca()

        # Handle optional argument for y labels.  If none exist, create defaults in the type of "Data Set 1", "Data Set 2" ...
        if yDataLabels is None:
            yDataLabels = []
            for i in range(1, len(yData)+1):
                yDataLabels.append("Data Set "+str(i))

        # Handle optional xData.  If none exist, create a set of integers from 1...N where N is the length of the y data.
        if xData is None:
            xData = range(1, len(yData[0])+1)

        # Convert the kwargs into individual series kwargs.
        seriesKeyWordArgs = PlotHelper.ConvertKeyWordArgumentsToSeriesSets(len(yDataLabels), **kwargs)

        # Need to repackage all the key word arguments.
        for dataSet, label, seriesKwargs in zip(yData, yDataLabels, seriesKeyWordArgs):
            axes.plot(xData, dataSet, label=label, **seriesKwargs)

        # Label the plot.
        AxesHelper.Label(axes, title=title, xLabels=xLabel, yLabels=yLabel)

        if showLegend:
            figure.legend(loc="upper left", bbox_to_anchor=(0, -0.12*PlotHelper.formatSettings.Scale), ncol=2, bbox_transform=axes.transAxes)

        if show:
            plt.show()

        return figure, axes


    @classmethod
    def NewMultiXAxesPlot(cls, data:pd.DataFrame, yAxisColumnName:str, axesesColumnNames:list, **kwargs):
        """
        Plots data on two axes with the same y-axis but different x-axis scales.

        Parameters
        ----------
        data : pandas.DataFrame
            The data.
        xAxisColumnName : string
            Independent variable column in the data.
        axesesColumnNames : array like of array like of strings
            Column names of the data to plot.  The array contains one set (array) of strings for the data to plot on
            each axes.  Example: [[column1, column2], [column3], [column 4, column5]] creates a three axes plot with
            column1 and column2 plotted on the left axes, column3 plotted on the first right axes, and column4 and column5
            plotted on the second right axes.
        **kwargs : keyword arguments
            These arguments are passed to the plot function.  Each keyword argument can be a single value or a list.  If it is
            a single value, the same value is used for every call to plat.  If it is a list, the values are passed in order to
            each series as it is plotted.
            Example 1:
                axesesColumnNames=['Column 1', 'Column 2'], linewidth=4
            Result
                The data in 'Column 1' and 'Column 2' are potted with a 'linewidth' of 4.
            Example 2:
                axesesColumnNames=['Column 1', ['Column 2', 'Column 3'], 'Column 4'], linewidth=[1, 2, 3, 4]
            Result
                The data in 'Column 1', 'Column 2', 'Column 3', and 'Column 4' are potted with a 'linewidth's of 1. 2. 3. and 4, respectively.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        axeses : tuple of matplotlib.axes.Axes
            The axes of the plot.
        """
        # Creates a figure with two axes having an aligned (shared) x-axis.
        figure, axeses    = PlotHelper.NewMultiXAxesFigure(len(axesesColumnNames))

        cls.PlotMultiXAxes(axeses, data, yAxisColumnName, axesesColumnNames, **kwargs)

        AxesHelper.AlignXAxes(axeses)

        return figure, axeses


    @classmethod
    def NewMultiYAxesPlot(cls, data:pd.DataFrame, xAxisColumnName:str, axesesColumnNames:list, **kwargs):
        """
        Plots data on two axes with the same x-axis but different y-axis scales.  The y-axis are on either side (left and right)
        of the plot.

        Parameters
        ----------
        data : pandas.DataFrame
            The data.
        xAxisColumnName : string
            Independent variable column in the data.
        axesesColumnNames : array like of array like of strings
            Column names of the data to plot.  The array contains one set (array) of strings for the data to plot on
            each axes.  Example: [[column1, column2], [column3], [column 4, column5]] creates a three axes plot with
            column1 and column2 plotted on the left axes, column3 plotted on the first right axes, and column4 and column5
            plotted on the second right axes.
        **kwargs : keyword arguments
            These arguments are passed to the plot function.  Each keyword argument can be a single value or a list.  If it is
            a single value, the same value is used for every call to plat.  If it is a list, the values are passed in order to
            each series as it is plotted.
            Example 1:
                axesesColumnNames=['Column 1', 'Column 2'], linewidth=4
            Result
                The data in 'Column 1' and 'Column 2' are potted with a 'linewidth' of 4.
            Example 2:
                axesesColumnNames=['Column 1', ['Column 2', 'Column 3'], 'Column 4'], linewidth=[1, 2, 3, 4]
            Result
                The data in 'Column 1', 'Column 2', 'Column 3', and 'Column 4' are potted with a 'linewidth's of 1. 2. 3. and 4, respectively.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        axeses : tuple of matplotlib.axes.Axes
            The axes of the plot.
        """
        # Creates a figure with two axes having an aligned (shared) x-axis.
        figure, axeses = PlotHelper.NewMultiYAxesFigure(len(axesesColumnNames))

        cls.PlotMultiYAxes(axeses, data, xAxisColumnName, axesesColumnNames, **kwargs)

        AxesHelper.AlignYAxes(axeses)

        return figure, axeses

    @classmethod
    def PlotMultiXAxes(cls, axeses:list, data:pd.DataFrame, yAxisColumnName:str, axesesColumnNames:list, **kwargs):
        """
        Plots data on two axes with the same x-axis but different y-axis scales.  The y-axis are on either side (left and right)
        of the plot.

        Parameters
        ----------
        axes : array like
            A an array of axes to plot on.  There should be one axes for each grouping (list/array) in axesesColumnNames.
        data : pandas.DataFrame
            The data.
        yAxisColumnName : string
            Independent variable column in the data.
        axesesColumnNames : array like of array like of strings
            Column names of the data to plot.  The array contains one set (array) of strings for the data to plot on
            each axes.  Example: [[column1, column2], [column3], [column 4, column5]] creates a three axes plot with
            column1 and column2 plotted on the left axes, column3 plotted on the first right axes, and column4 and column5
            plotted on the second right axes.
        **kwargs : keyword arguments
            These arguments are passed to the plot function.  Each keyword argument can be a single value or a list.  If it is
            a single value, the same value is used for every call to plat.  If it is a list, the values are passed in order to
            each series as it is plotted.
            Example 1:
                axesesColumnNames=['Column 1', 'Column 2'], linewidth=4
            Result
                The data in 'Column 1' and 'Column 2' are potted with a 'linewidth' of 4.
            Example 2:
                axesesColumnNames=['Column 1', ['Column 2', 'Column 3'], 'Column 4'], linewidth=[1, 2, 3, 4]
            Result
                The data in 'Column 1', 'Column 2', 'Column 3', and 'Column 4' are potted with a 'linewidth's of 1. 2. 3. and 4, respectively.

        Returns
        -------
        lines2d : list of Line2D
            The plotted line objects.
        """
        cls._PlotMultiAxes(axeses, data, yAxisColumnName, axesesColumnNames, "y", **kwargs)


    @classmethod
    def PlotMultiYAxes(cls, axeses:list, data:pd.DataFrame, xAxisColumnName:str, axesesColumnNames:list, **kwargs):
        """
        Plots data on two axes with the same x-axis but different y-axis scales.  The y-axis are on either side (left and right)
        of the plot.

        Parameters
        ----------
        axes : array like
            A an array of axes to plot on.  There should be one axes for each grouping (list/array) in axesesColumnNames.
        data : pandas.DataFrame
            The data.
        xAxisColumnName : string
            Independent variable column in the data.
        axesesColumnNames : array like of array like of strings
            Column names of the data to plot.  The array contains one set (array) of strings for the data to plot on
            each axes.  Example: [[column1, column2], [column3], [column 4, column5]] creates a three axes plot with
            column1 and column2 plotted on the left axes, column3 plotted on the first right axes, and column4 and column5
            plotted on the second right axes.
        **kwargs : keyword arguments
            These arguments are passed to the plot function.  Each keyword argument can be a single value or a list.  If it is
            a single value, the same value is used for every call to plat.  If it is a list, the values are passed in order to
            each series as it is plotted.
            Example 1:
                axesesColumnNames=['Column 1', 'Column 2'], linewidth=4
            Result
                The data in 'Column 1' and 'Column 2' are potted with a 'linewidth' of 4.
            Example 2:
                axesesColumnNames=['Column 1', ['Column 2', 'Column 3'], 'Column 4'], linewidth=[1, 2, 3, 4]
            Result
                The data in 'Column 1', 'Column 2', 'Column 3', and 'Column 4' are potted with a 'linewidth's of 1. 2. 3. and 4, respectively.

        Returns
        -------
        lines2d : list of Line2D
            The plotted line objects.
        """
        cls._PlotMultiAxes(axeses, data, xAxisColumnName, axesesColumnNames, "x", **kwargs)


    @classmethod
    def _PlotMultiAxes(cls, axeses:list, data:pd.DataFrame, independentColumnName:str, axesesColumnNames:list, independentAxis:str, **kwargs):
        """
        Plots data on two axes with the same x-axis but different y-axis scales.  The y-axis are on either side (left and right)
        of the plot.

        Parameters
        ----------
        axes : array like
            A an array of axes to plot on.  There should be one axes for each grouping (list/array) in axesesColumnNames.
        data : pandas.DataFrame
            The data.
        independentColumnName : string
            Independent variable column in the data.
        axesesColumnNames : array like of array like of strings
            Column names of the data to plot.  The array contains one set (array) of strings for the data to plot on
            each axes.  Example: [[column1, column2], [column3], [column 4, column5]] creates a three axes plot with
            column1 and column2 plotted on the left axes, column3 plotted on the first right axes, and column4 and column5
            plotted on the second right axes.
        independentAxis : str
            Which axis is independent.
        **kwargs : keyword arguments
            These arguments are passed to the plot function.  Each keyword argument can be a single value or a list.  If it is
            a single value, the same value is used for every call to plat.  If it is a list, the values are passed in order to
            each series as it is plotted.
            Example 1:
                axesesColumnNames=['Column 1', 'Column 2'], linewidth=4
            Result
                The data in 'Column 1' and 'Column 2' are potted with a 'linewidth' of 4.
            Example 2:
                axesesColumnNames=['Column 1', ['Column 2', 'Column 3'], 'Column 4'], linewidth=[1, 2, 3, 4]
            Result
                The data in 'Column 1', 'Column 2', 'Column 3', and 'Column 4' are potted with a 'linewidth's of 1. 2. 3. and 4, respectively.

        Returns
        -------
        lines2d : list of Line2D
            The plotted line objects.
        """
        # The colors are needed because each axes wants to use it's own color cycle resulting in duplication of
        # colors on the two axes.  Therefore, we have to manually specify the colors so they don't repeat.  This is
        # done by using the PlotHelper.NextColor() function.

        # Convert the kwargs into individual series kwargs.
        seriesKeyWordArgs = PlotHelper.ConvertKeyWordArgumentsToSeriesSets(ListTools.GetLengthOfNestedObjects(axesesColumnNames), **kwargs)

        # Store the plotted lines so we can return them.
        lines2d     = []
        seriesIndex = 0

        for axesColumnNames, axes in zip(axesesColumnNames, axeses):
            for column in axesColumnNames:

                # Key word arguments.  Start with a default set and then override with any specified as arguments.
                defaultKwargs = {"color" : PlotHelper.NextColor(), "label" : column}
                defaultKwargs.update(seriesKeyWordArgs[seriesIndex])

                if independentAxis == "x":
                    # X-axis is independent axis.
                    lines = axes.plot(data[independentColumnName], data[column], **defaultKwargs)
                else:
                    # Y-axis is independent axis.
                    lines = axes.plot(data[column], data[independentColumnName], **defaultKwargs)

                lines2d.append(lines[0])
                seriesIndex += 1
        return lines2d


    @classmethod
    def CreateConfusionMatrixPlot(cls, confusionMatrix, title, titleSuffix=None, axesLabels=None):
        """
        Plots the confusion matrix for the model output.

        Parameters
        ----------
        confusionMatrix : ndarray of shape (n_classes, n_classes)
        title : string
            Main title for the data.
        titleSuffix : string or None, optional
            If supplied, the string is prepended to the title.
        axesLabels : array like of strings
            Labels to use on the predicted and actual axes.

        Returns
        -------
        figure : Figure
            The newly created figure.
        """
        numberOfCategories = confusionMatrix.shape[0]

        if numberOfCategories != confusionMatrix.shape[1]:
            raise Exception("The confusion matrix supplied is not square.")

        # The numpy array has to be set as an object type.  If set (or allowed to assume) a type of "str" the entry is created
        # only large enough for the initial string (a character type is used).  It is not possible to append to it.
        labels = np.asarray(
            [
                ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item/confusionMatrix.flatten().sum())]
                for item in confusionMatrix.flatten()
            ]
        ).astype("object").reshape(numberOfCategories, numberOfCategories)

        # Tack on the type labels to the numerical information.
        if numberOfCategories == 2:
            labels[0, 0] += "\nTN"
            labels[1, 0] += "\nFN\nType 2"
            labels[0, 1] += "\nFP\nType 1"
            labels[1, 1] += "\nTP"

        # Must be run before creating figure or plotting data.
        # The standard scale for this plot will be a little higher than the normal scale.
        # Not much is shown, so we can shrink the figure size.
        categorySizeAdjustment = 0.65*(numberOfCategories-2)
        PlotHelper.Format()

        # Create plot.
        figure = plt.gcf()

        # Set the figure size.
        figure.set_figwidth(5.35+categorySizeAdjustment)
        figure.set_figheight(4+categorySizeAdjustment)

        axes   = sns.heatmap(confusionMatrix, cmap=PlotMaker.colorMap, annot=labels, annot_kws={"fontsize" : 12*PlotHelper.formatSettings.Scale}, fmt="")
        AxesHelper.Label(axes, title=title, xLabels="Predicted", yLabels="Actual", titleSuffix=titleSuffix)

        if axesLabels is not None:
            axes.xaxis.set_ticklabels(axesLabels, rotation=90)
            axes.yaxis.set_ticklabels(axesLabels, rotation=0)

        # Turn off the grid.
        axes.grid(False)

        plt.show()

        return figure


    @classmethod
    def CreateColorCyclePlot(cls, lineColorCycle=None):
        """
        Create a plot that shows the colors in a color cycle.

        Parameters
        ----------
        lineColorCycle : string, optional
            The color cycle to plot. The default is None.  These can be any color cycle accepted by PlotHelper.

        Returns
        -------
        None.
        """
        PlotHelper.Format()

        numberOfPoints  = 5
        figure, axes    = plt.subplots()
        x               = range(numberOfPoints)
        colors          = PlotHelper.GetColorCycle(lineColorCycle=lineColorCycle, numberFormat="hex")
        numberOfColors  = len(colors)

        # Turn off the bounding box (set of splines that surround axes).
        [spline.set_visible(False) for spline in axes.spines.values()]


        for i in range(numberOfColors):
            # Set the y to the same as the position in the color cycle.
            y = [i] * numberOfPoints
            axes.plot(x, y, label="data", marker="o", markerfacecolor=colors[i], markeredgecolor=colors[i], markeredgewidth=10, markersize=20, linewidth=10, color=colors[i])

            # Plot the name on the right.  The location is the x, y plot point adjusted to center it.
            plt.annotate(str(colors[i]), (numberOfPoints-0.75, i+0.15), annotation_clip=False)

        # Display the name of the color cycle.
        axes.xaxis.label.set_fontsize(40)
        if lineColorCycle == None:
            axes.set(xlabel=PlotHelper.lineColorCycle)
        else:
            axes.set(xlabel=lineColorCycle)

        # Reverse the y-axis so that the lower numbers are on top.
        AxesHelper.SetYAxisLimits(axes, limits=[-1, 10], numberOfTicks=numberOfPoints+2)
        AxesHelper.ReverseYAxisLimits(axes)

        # Clear the x axis labels and use the y axis labels to label the position of the color.
        plt.xticks([])
        plt.yticks(range(numberOfColors))

        plt.show()


    @classmethod
    def ApplyPlotToEachCategory(cls, data, columns, plotFunction, save=False, **kwargs):
        """
        Creates a new figure for every entry in the list of columns.

        Parameters
        ----------
        data : Pandas DataFrame
            The data.
        columns : an arry or list of strings
            Column names in the DataFrame.
        plotFunction : function
            Plotting function to apply to all columns.
        save : bool
            If true, the plots are saved to the default plotting directory.
        **kwargs : keyword arguments
            These arguments are passed on to the plotFunction.

        Returns
        -------
        None.
        """
        for column in columns:
            figure = plotFunction(data, column, **kwargs)

            if save:
                fileName = plotFunction.__name__ + column.title() + " Category"
                cls.SavePlot(fileName, figure=figure)