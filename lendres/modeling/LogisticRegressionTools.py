"""
Created on January 19, 2022
@author: Lance A. Endres
"""
import pandas                                                        as pd
import numpy                                                         as np
import matplotlib.pyplot                                             as plt

from sklearn                                                         import metrics

from   lendres.plotting.AxesHelper                                   import AxesHelper
from   lendres.plotting.PlotHelper                                   import PlotHelper

class LogisticRegressionTools():


    @classmethod
    def GetRocCurveAndScores(cls, y, yPredictedProbabilities):
        """
        Calculates the area under the curve and geomtric mean.

        Parameters
        ----------
        y : array like of ints
            True values.
        yPredicted : array like of doubles
            Predicted probabilities.
        Returns
        -------
        falsePositiveRates : array of doubles
            False positive rates (range 0-1).
        truePositiveRates : array of doubles
            True positive rates (range 0-1).
        scores : pandas.DataFrame
            DataFrame in the format:
                area_under_curve   best_geometric_mean    best_threshold   best_threshold_index
        """
        # Get the area under the curve score.
        aucScore                                          = metrics.roc_auc_score(y, yPredictedProbabilities)
        falsePositiveRates, truePositiveRates, thresholds = metrics.roc_curve(y, yPredictedProbabilities)

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
        geometricMeans = np.sqrt(truePositiveRates * (1-falsePositiveRates))

        # Locate the index of the largest geometric mean.
        index = np.argmax(geometricMeans)

        scores  = pd.DataFrame(
            [[aucScore, geometricMeans[index], thresholds[index], index]],
            index=["Best"],
            columns=["Area Under Curve", "Best Geometric Mean", "Best Threshold", "Index of Best Threshold"]
        )

        return falsePositiveRates, truePositiveRates, scores


    @classmethod
    def CreateRocCurvePlot(cls, dataSets, titleSuffix=None, **kwargs):
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
            cls.PlotRocCurve(value[0], value[1], key)

        # Plot the diagonal line, the wrost fit possible line.
        plt.plot([0, 1], [0, 1], "r--")

        # Formatting the axes.
        figure = plt.gcf()
        axes   = plt.gca()
        title  = "Receiver Operating Characteristic"

        AxesHelper.Label(axes, title=title, xLabels="False Positive Rate", yLabels="True Positive Rate", titleSuffix=titleSuffix)
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