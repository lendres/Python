"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import pandas                                                             as pd
import numpy                                                              as np
import matplotlib.pyplot                                                  as plt

import seaborn                                                            as sns
sns.set(color_codes=True)

from   lendres.plotting.AxesHelper                                        import AxesHelper
from   lendres.plotting.PlotHelper                                        import PlotHelper
from   lendres.LogisticRegressionTools                                    import LogisticRegressionTools
from   lendres.algorithms.DataType                                        import DataType

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
        AxesHelper.Label(axes, title=title, xLabel=xLabel, yLabels=yLabel)

        if showLegend:
            figure.legend(loc="upper left", bbox_to_anchor=(0, -0.12*PlotHelper.scale), ncol=2, bbox_transform=axes.transAxes)

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
            These arguments are passed to the plot function.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        axeses : tuple of matplotlib.axes.Axes
            The axes of the plot.
        """
        # Creates a figure with two axes having an aligned (shared) x-axis.
        figure, axeses    = PlotHelper.NewMultiXAxesFigure(len(axesesColumnNames))

        cls.MultiAxesPlot(axeses, data, yAxisColumnName, axesesColumnNames, "y", **kwargs)

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
            These arguments are passed to the plot function.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        axeses : tuple of matplotlib.axes.Axes
            The axes of the plot.
        """
        # Creates a figure with two axes having an aligned (shared) x-axis.
        figure, axeses = PlotHelper.NewMultiYAxesFigure(len(axesesColumnNames))

        cls.MultiAxesPlot(axeses, data, xAxisColumnName, axesesColumnNames, "x", **kwargs)

        AxesHelper.AlignYAxes(axeses)

        return figure, axeses


    @classmethod
    def MultiAxesPlot(cls, axeses:list, data:pd.DataFrame, independentColumnName:str, axesesColumnNames:list, independentAxis:str="x", **kwargs):
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
        **kwargs : keyword arguments
            These arguments are passed to the plot function.

        Returns
        -------
        None.
        """
        # The colors are needed because each axes wants to use it's own color cycle resulting in duplication of
        # colors on the two axes.  Therefore, we have to manually specify the colors so they don't repeat.  This is
        # done by using the PlotHelper.NextColor() function.

        # Convert the kwargs into individual series kwargs.
        seriesKeyWordArgs = PlotHelper.ConvertKeyWordArgumentsToSeriesSets(DataType.GetLengthOfNestedObjects(axesesColumnNames), **kwargs)

        # Store the plotted lines so we can return them.
        lines2d     = []
        seriesIndex = 0

        for axesColumnNames, axes in zip(axesesColumnNames, axeses):
            for column in axesColumnNames:

                # Key word arguments.  Start with a default set and then override with any specified as arguments.
                defaultKwargs = {"color" : PlotHelper.NextColor()}
                seriesKwargs  = seriesKeyWordArgs[seriesIndex]
                defaultKwargs.update(seriesKwargs)

                if independentAxis == "x":
                    lines = axes.plot(data[independentColumnName], data[column], label=column, **defaultKwargs)
                else:
                    lines = axes.plot(data[column], data[independentColumnName], label=column, **defaultKwargs)

                lines2d.append(lines[0])
                seriesIndex += 1
        return lines2d


    @classmethod
    def CreateCountFigure(cls, data, primaryColumnName, subColumnName=None, titleSuffix=None, xLabelRotation=None):
        """
        Creates a bar chart that plots a primary category and subcategory as the hue.

        Parameters
        ----------
        data : Pandas DataFrame
            The data.
        primaryColumnName : string
            Column name in the DataFrame.
        subColumnName : string
            If present, the column used as the hue.
        titleSuffix : string or None, optional
            If supplied, the string is prepended to the title.
        xLabelRotation : float
            Rotation of x labels.

        Returns
        -------
        figure : Figure
            The newly created figure.
        """
        # Must be run before creating figure or plotting data.
        PlotHelper.Format()

        # This creates the bar chart.  At the same time, save the figure so we can return it.
        axes = sns.countplot(x=primaryColumnName, data=data, hue=subColumnName)
        figure = plt.gcf()

        # Label the perentages of each column.
        cls.LabelPercentagesOnColumnsOfBarGraph(axes)

        # If adding a hue, set the legend to run horizontally.
        if subColumnName is not None:
            ncol = data[subColumnName].nunique()
            plt.legend(loc="upper right", borderaxespad=0, ncol=ncol)

        # Titles.
        title = "\"" + primaryColumnName + "\"" + " Category"
        AxesHelper.Label(axes, title=title, xLabel=subColumnName, yLabels="Count", titleSuffix=titleSuffix)

        # Option to rotate the x-axis labels.
        AxesHelper.RotateXLabels(xLabelRotation)

        # Make sure the plot is shown.
        plt.show()

        return figure


    @classmethod
    def LabelPercentagesOnColumnsOfBarGraph(cls, axes):
        """
        Labels each column with a percentage of the total sum of all columns.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            Matplotlib axes to plot on.

        Returns
        -------
        None.
        """
        # Number of entries.
        total = 0

        # Find the total count first.
        for patch in axes.patches:
            total += patch.get_height()

        for patch in axes.patches:
            # Percentage of the column.
            percentage = "{:.1f}%".format(100*patch.get_height()/total)

            # Find the center of the column/patch on the x-axis.
            x = patch.get_x() + patch.get_width()/2

            # Height of the column/patch.  Add a little so it does not touch the top of the column.
            y = patch.get_y() + patch.get_height() + 0.5

            # Plot a label slightly above the column and use the horizontal alignment to center it in the column.
            axes.annotate(percentage, (x, y), size=PlotHelper.GetScaledAnnotationSize(), fontweight="bold", horizontalalignment="center")


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
        PlotHelper.Format(parameterFile="gridless.mplstyle", width=5.35+categorySizeAdjustment, height=4+categorySizeAdjustment)

        # Create plot and set the titles.
        axes = sns.heatmap(confusionMatrix, cmap=PlotMaker.colorMap, annot=labels, annot_kws={"fontsize" : 12*PlotHelper.scale}, fmt="")
        AxesHelper.Label(axes, title=title, xLabel="Predicted", yLabels="Actual", titleSuffix=titleSuffix)

        if axesLabels is not None:
            axes.xaxis.set_ticklabels(axesLabels, rotation=90)
            axes.yaxis.set_ticklabels(axesLabels, rotation=0)

        figure = plt.gcf()
        plt.show()

        return figure


    @classmethod
    def CreateRocCurvePlot(self, dataSets, titleSuffix=None, **kwargs):
        """
        Creates a plot of the receiver operatoring characteristic curve(s).

        Parameters
        ----------
        dataSets : dictionary
            Data set(s) to plot.
            The key is one of:
                training - Labels and colors the data as training data.
                validation - Labels and colors the data as validation data.
                testing  - Labels and colors the data as testing data.
            The values are of the form [trueValue, predictedValues]
        **kwargs :  keyword arguments
            keyword arguments pass on to the plot formating function.

        Returns
        -------
        figure : matplotlib.pyplot.figure
            The newly created figure.
        axes : matplotlib.axes.Axes
            The axes of the plot.
        """
        # Must be run before creating figure or plotting data.
        PlotHelper.Format(**kwargs)

        # Plot the ROC curve(s).
        for key, value in dataSets.items():
            PlotMaker.PlotRocCurve(value[0], value[1], key)

        # Plot the diagonal line, the wrost fit possible line.
        plt.plot([0, 1], [0, 1], "r--")

        # Formatting the axes.
        figure = plt.gcf()
        axes   = plt.gca()
        title  = "Receiver Operating Characteristic"

        AxesHelper.Label(axes, title=title, xLabel="False Positive Rate", yLabels="True Positive Rate", titleSuffix=titleSuffix)
        axes.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05])

        plt.legend(loc="lower right")
        plt.show()

        return figure, axes


    @classmethod
    def PlotRocCurve(cls, y, yPredicted, which):
        """
        Plots the receiver operatoring characteristic curve.

        Parameters
        ----------
        y : array
            True values.
        yPredicted : array
            Predicted values.
        which : string
            Which data set is being plotted.
            training - Labels and colors the data as training data.
            validation - Labels and colors the data as validation data.
            testing  - Labels and colors the data as testing data.

        Returns
        -------
        None.
        """
        color = None
        if which == "training":
            color = "#1f77b4"
        elif which == "validation":
            color = "#a55af4"
        elif which == "testing":
            color = "#ff7f0e"
        else:
            raise Exception("Invalid data set specified for the which parameter.")

        # Get values for plotting the curve and the scores associated with the curve.
        falsePositiveRates, truePositiveRates, scores = LogisticRegressionTools.GetRocCurveAndScores(y, yPredicted)

        label = which.title()+" (area = %0.2f)" % (scores["Area Under Curve"]).iloc[0]
        plt.plot(falsePositiveRates, truePositiveRates, label=label, color=color)


        index = (scores["Index of Best Threshold"]).iloc[0]
        label = which.title() + " Best Threshold %0.3f" % (scores["Best Threshold"]).iloc[0]
        plt.scatter(falsePositiveRates[index], truePositiveRates[index], marker="o", color=color, label=label)


    @classmethod
    def PlotColorCycle(cls, lineColorCycle=None):
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
        PlotHelper.Format(formatStyle="pyplot")

        numberOfPoints  = 5
        figure, axes    = plt.subplots()
        x               = range(numberOfPoints)
        colors          = PlotHelper.GetColorCycle(lineColorCycle=lineColorCycle, numberFormat="hex")
        numberOfColors  = len(colors)

        for i in range(numberOfColors):
            # Set the y to the same as the position in the color cycle.
            y = [i] * numberOfPoints
            axes.plot(x, y, label="data", marker="o", markerfacecolor=colors[i], markeredgecolor=colors[i], markeredgewidth=10, markersize=20, linewidth=10, color=colors[i])

            # Plot the name on the right.
            plt.annotate(str(colors[i]), (numberOfPoints-0.75, i-0.15), annotation_clip=False)

            # Clear the x axis labels and use the y axis labels to label the position of the color.
            plt.xticks([])
            plt.yticks(range(numberOfColors))

            # Turn off the bounding box (spline).
            [spline.set_visible(False) for spline in axes.spines.values()]

            # Display the name of the color cycle.
            axes.xaxis.label.set_fontsize(40)
            if lineColorCycle == None:
                axes.set(xlabel=PlotHelper.lineColorCycle)
            else:
                axes.set(xlabel=lineColorCycle)
        plt.show()