"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import numpy                                     as np
import matplotlib.pyplot                         as plt

import seaborn                                   as sns
sns.set(color_codes=True)

from   lendres.PlotHelper                        import PlotHelper
from   lendres.LogisticRegressionTools           import LogisticRegressionTools

class PlotMaker():
    # Class level variables.

    # Color map to use for plots.
    colorMap      = None


    @classmethod
    def CreateCountPlot(cls, data, primaryColumnName, subColumnName=None, titlePrefix=None, xLabelRotation=None):
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
        cls.LabelPercentagesOnCountPlot(axis)

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
    def LabelPercentagesOnCountPlot(cls, axis):
        """
        Plot the percentages of each entry of a column.

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

            # Hieght of the column/patch.  Add a little so it does not touch the top of the column.
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