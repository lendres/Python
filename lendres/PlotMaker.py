"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import numpy                                as np
import matplotlib.pyplot                    as plt

import seaborn                              as sns
sns.set(color_codes=True)

from sklearn                                import metrics


from lendres.PlotHelper                     import PlotHelper

class PlotMaker():
    # Class level variables.

    # Color map to use for plots.
    colorMap      = None

    @classmethod
    def CreateConfusionMatrixPlot(cls, confusionMatrix, title, titlePrefix=None):
        """
        Plots the confusion matrix for the model output.

        Parameters
        ----------
        confusionMatrix : ndarray of shape (n_classes, n_classes)
        title : string
            Main title for the data.
        titlePrefix : string or None, optional
            If supplied, the string is prepended to the title.

        Returns
        -------
        confusionMatrix : ndarray of shape (n_classes, n_classes)
            The confusion matrix for the data.
        """
        numberOfCategories = confusionMatrix.shape[0]

        if numberOfCategories != confusionMatrix.shape[1]:
            raise Exception("The confusion matrix supplied is not square.")

        # The numpy array has to be set as an object type.  If set (or allowed to assume) a
        # type of "str" the entry is created only large enough for the initial string (a character
        # type is used).  It is not possible to append to it.
        labels = np.asarray(
            [
                ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / confusionMatrix.flatten().sum())]
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
        PlotHelper.FormatPlot(width=5.35, height=4)

        # Create plot and set the titles.
        axis = sns.heatmap(confusionMatrix, cmap=PlotMaker.colorMap, annot=labels, annot_kws={"fontsize" : 14*PlotHelper.scale}, fmt="")
        PlotHelper.Label(axis, title=title, xLabel="Predicted", yLabel="Actual", titlePrefix=titlePrefix)

        plt.show()

        return confusionMatrix


    @classmethod
    def CreateRocCurvePlot(self, dataSets, titlePrefix, **kwargs):
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

        # Get the area under the curve score.
        aucScore                                          = metrics.roc_auc_score(y, yPredicted)
        falsePositiveRates, truePositiveRates, thresholds = metrics.roc_curve(y, yPredicted)


        plt.plot(falsePositiveRates, truePositiveRates, label=which.title()+" (area = %0.2f)"%aucScore, color=color)

        # Calculate the geometric mean for each threshold.
        #
        # The true positive rate is called the Sensitivity. The inverse of the false-positive rate is called the Specificity.
        # Sensitivity = True Positive / (True Positive + False Negative)
        # Specificity = True Negative / (False Positive + True Negative)
        # Where:
        #     Sensitivity = True Positive Rate
        #     Specificity = 1 – False Positive Rate
        #
        # The geometric mean is a metric for imbalanced classification that, if optimized, will seek a balance between the
        # sensitivity and the specificity.

        # geometric mean = sqrt(Sensitivity * Specificity)
        gmeans = np.sqrt(truePositiveRates * (1-falsePositiveRates))

        # Locate the index of the largest geometric mean.
        index = np.argmax(gmeans)
        print('Best Threshold=%f, Geometric Mean=%.3f' % (thresholds[index], gmeans[index]))

        plt.scatter(falsePositiveRates[index], truePositiveRates[index], marker="o", color=color, label=which.title()+" Best Threshold %0.3f"%thresholds[index])