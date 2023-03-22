"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import numpy                                     as np
import matplotlib.pyplot                         as plt

import seaborn                                   as sns
sns.set(color_codes=True)

from   lendres.plotting.PlotHelper               import PlotHelper
from   lendres.LogisticRegressionTools           import LogisticRegressionTools

class PlotMaker():
    # Class level variables.

    # Color map to use for plots.
    colorMap      = None

    @classmethod
    def CreateFastFigure(cls, yData, yDataLabels=None, xData=None, title=None, xAxisLabe=None, yAxisLabel=None):
        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        figure = plt.gcf()
        axis   = plt.gca()


        # Handle optional argument for y labels.  If none exist, create defaults in the type of "Data Set 1", "Data Set 2" ...
        if yDataLabels is None:
            yDataLabels = []
            for i in range(1, len(yData)+1):
                yDataLabels.append("Data Set "+str(i))


        # Handle optional xData.  If none exist, create a set of integers from 1...N where N is the length of the y data.
        if xData is None:
            xData = range(1, len(yData[0])+1)

        # Plot all the data sets.
        for dataSet, label in zip(yData, yDataLabels):
            axis.plot(xData, dataSet, label=label)

        # Label the plot.
        PlotHelper.Label(axis, title=title, xLabel=xAxisLabe, yLabel=yAxisLabel)

        axis.grid()
        figure.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=2, bbox_transform=axis.transAxes)
        plt.show()

        return figure, axis


    @classmethod
    def NewMultiXAxesPlot(cls, data, yAxisColumnName, axesColumnNames, colorCycle=None, **kwargs):
        """
        Plots data on two axes with the same x-axis but different y-axis scales.  The y-axis are on either side (left and right)
        of the plot.

        Parameters
        ----------
        data : pandas.DataFrame
            The data.
        xAxisColumnName : string
            Independent variable column in the data.
        axesColumnNames : array like of array like of strings
            Column names of the data to plot.  The array contains one set (array) of strings for the data to plot on
            each axis.  Example: [[column1, column2], [column3], [column 4, column5]] creates a three axes plot with
            column1 and column2 plotted on the left axis, column3 plotted on the first right axis, and column4 and column5
            plotted on the second right axis.
       colorCycle : array like, optional
            The colors to use for the plotted lines. The default is None.
        **kwargs : keyword arguments
            These arguments are passed to the plot function.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        axis : tuple of matplotlib.pyplot.axis
            The axes of the plot.
        """
        # Creates a figure with two axes having an aligned (shared) x-axis.
        figure, axes    = PlotHelper.NewMultiXAxesFigure(len(axesColumnNames))

        cls.MultiAxesPlot(axes, data, yAxisColumnName, axesColumnNames, "y", colorCycle=None, **kwargs)

        #PlotHelper.AlignXAxes(axes)

        return figure, axes


    @classmethod
    def NewMultiYAxesPlot(cls, data, xAxisColumnName, axesColumnNames, colorCycle=None, **kwargs):
        """
        Plots data on two axes with the same x-axis but different y-axis scales.  The y-axis are on either side (left and right)
        of the plot.

        Parameters
        ----------
        data : pandas.DataFrame
            The data.
        xAxisColumnName : string
            Independent variable column in the data.
        axesColumnNames : array like of array like of strings
            Column names of the data to plot.  The array contains one set (array) of strings for the data to plot on
            each axis.  Example: [[column1, column2], [column3], [column 4, column5]] creates a three axes plot with
            column1 and column2 plotted on the left axis, column3 plotted on the first right axis, and column4 and column5
            plotted on the second right axis.
       colorCycle : array like, optional
            The colors to use for the plotted lines. The default is None.
        **kwargs : keyword arguments
            These arguments are passed to the plot function.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        axis : tuple of matplotlib.pyplot.axis
            The axes of the plot.
        """
        # Creates a figure with two axes having an aligned (shared) x-axis.
        figure, axes    = PlotHelper.NewMultiYAxesFigure(len(axesColumnNames))

        cls.MultiAxesPlot(axes, data, xAxisColumnName, axesColumnNames, "x", colorCycle=None, **kwargs)

        PlotHelper.AlignYAxes(axes)

        return figure, axes


    @classmethod
    def MultiAxesPlot(cls, axes, data, independentColumnName, axesColumnNames, independentAxis, colorCycle=None, **kwargs):
        """
        Plots data on two axes with the same x-axis but different y-axis scales.  The y-axis are on either side (left and right)
        of the plot.

        Parameters
        ----------
        axes : array like
            A an array of axes to plot on.  There should be one axis for each grouping (list/array) in axesColumnNames.
        data : pandas.DataFrame
            The data.
        independentColumnName : string
            Independent variable column in the data.
        axesColumnNames : array like of array like of strings
            Column names of the data to plot.  The array contains one set (array) of strings for the data to plot on
            each axis.  Example: [[column1, column2], [column3], [column 4, column5]] creates a three axes plot with
            column1 and column2 plotted on the left axis, column3 plotted on the first right axis, and column4 and column5
            plotted on the second right axis.
       colorCycle : array like, optional
            The colors to use for the plotted lines. The default is None.
        **kwargs : keyword arguments
            These arguments are passed to the plot function.

        Returns
        -------
        None.
        """
        # The colors are needed because each axis wants to use it's own color cycle resulting in duplication of
        # colors on the two axis.  Therefore, we have to manually specify the colors so they don't repeat.
        if colorCycle is None:
            colorCycle = PlotHelper.GetColorCycle()
        color  = 0

        independentData = data[independentColumnName]

        for axisColumnNames, axis in zip(axesColumnNames, axes):
            for column in axisColumnNames:
                if independentAxis == "x":
                    axis.plot(independentData, data[column], color=colorCycle[color], label=column, **kwargs)
                else:
                    pass
                    axis.plot(data[column], independentData, color=colorCycle[color], label=column, **kwargs)
                color += 1

        axes[0].grid()


    @classmethod
    def CreateCountFigure(cls, data, primaryColumnName, subColumnName=None, titlePrefix=None, xLabelRotation=None):
        """
        Creates a bar chart that plots a primary category and subcategory as the  hue.

        Parameters
        ----------
        data : Pandas DataFrame
            The data.
        primaryColumnName : string
            Column name in the DataFrame.
        subColumnName : string
            If present, the column used as the hue.
        titlePrefix : string or None, optional
            If supplied, the string is prepended to the title.
        xLabelRotation : float
            Rotation of x labels.

        Returns
        -------
        figure : Figure
            The newly created figure.
        """
        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        # This creates the bar chart.  At the same time, save the figure so we can return it.
        axis = sns.countplot(x=primaryColumnName, data=data, hue=subColumnName)
        figure = plt.gcf()

        # Label the perentages of each column.
        cls.LabelPercentagesOnColumnsOfBarGraph(axis)

        # If adding a hue, set the legend to run horizontally.
        if subColumnName is not None:
            ncol = data[subColumnName].nunique()
            plt.legend(loc="upper right", borderaxespad=0, ncol=ncol)

        # Titles.
        title = "\"" + primaryColumnName + "\"" + " Category"
        PlotHelper.Label(axis, title=title, xLabel=subColumnName, yLabel="Count", titlePrefix=titlePrefix)

        # Option to rotate the x axis labels.
        PlotHelper.RotateXLabels(xLabelRotation)

        # Make sure the plot is shown.
        plt.show()

        return figure


    @classmethod
    def LabelPercentagesOnColumnsOfBarGraph(cls, axis):
        """
        Labels each column with a percentage of the total sum of all columns.

        Parameters
        ----------
        axis : axis
            Matplotlib axis to plot on.

        Returns
        -------
        None.
        """
        # Number of entries.
        total = 0

        # Find the total count first.
        for patch in axis.patches:
            total += patch.get_height()

        for patch in axis.patches:
            # Percentage of the column.
            percentage = "{:.1f}%".format(100*patch.get_height()/total)

            # Find the center of the column/patch on the x-axis.
            x = patch.get_x() + patch.get_width()/2

            # Height of the column/patch.  Add a little so it does not touch the top of the column.
            y = patch.get_y() + patch.get_height() + 0.5

            # Plot a label slightly above the column and use the horizontal alignment to center it in the column.
            axis.annotate(percentage, (x, y), size=PlotHelper.GetScaledAnnotationSize(), fontweight="bold", horizontalalignment="center")


    @classmethod
    def CreateConfusionMatrixPlot(cls, confusionMatrix, title, titlePrefix=None, axisLabels=None):
        """
        Plots the confusion matrix for the model output.

        Parameters
        ----------
        confusionMatrix : ndarray of shape (n_classes, n_classes)
        title : string
            Main title for the data.
        titlePrefix : string or None, optional
            If supplied, the string is prepended to the title.
        axisLabels : array like of strings
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
        PlotHelper.FormatPlot(width=5.35+categorySizeAdjustment, height=4+categorySizeAdjustment)

        # Create plot and set the titles.
        axis = sns.heatmap(confusionMatrix, cmap=PlotMaker.colorMap, annot=labels, annot_kws={"fontsize" : 12*PlotHelper.scale}, fmt="")
        PlotHelper.Label(axis, title=title, xLabel="Predicted", yLabel="Actual", titlePrefix=titlePrefix)

        if axisLabels is not None:
            axis.xaxis.set_ticklabels(axisLabels, rotation=90)
            axis.yaxis.set_ticklabels(axisLabels, rotation=0)

        figure = plt.gcf()
        plt.show()

        return figure


    @classmethod
    def CreateRocCurvePlot(self, dataSets, titlePrefix=None, **kwargs):
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
        axis : matplotlib.pyplot.axis
            The axis of the plot.
        """
        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot(**kwargs)

        # Plot the ROC curve(s).
        for key, value in dataSets.items():
            PlotMaker.PlotRocCurve(value[0], value[1], key)

        # Plot the diagonal line, the wrost fit possible line.
        plt.plot([0, 1], [0, 1], "r--")

        # Formatting the axis.
        figure = plt.gcf()
        axis   = plt.gca()
        title  = "Receiver Operating Characteristic"

        PlotHelper.Label(axis, title=title, xLabel="False Positive Rate", yLabel="True Positive Rate", titlePrefix=titlePrefix)
        axis.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05])

        plt.legend(loc="lower right")
        plt.show()

        return figure, axis


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

        label = which.title()+" (area = %0.2f)" % scores["Area Under Curve"]
        plt.plot(falsePositiveRates, truePositiveRates, label=label, color=color)


        index = scores["Index of Best Threshold"]
        label = which.title() + " Best Threshold %0.3f" % scores["Best Threshold"]
        plt.scatter(falsePositiveRates[index], truePositiveRates[index], marker="o", color=color, label=label)