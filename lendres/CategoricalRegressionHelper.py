"""
Created on January 19, 2022
@author: Lance A. Endres
"""
import pandas                                                   as pd
import matplotlib.pyplot                                        as plt
from   sklearn                                                  import metrics

from   lendres.CategoricalHelper                                import CategoricalHelper
from   lendres.plotting.AxesHelper                              import AxesHelper
from   lendres.plotting.PlotHelper                              import PlotHelper
from   lendres.plotting.PlotMaker                               import PlotMaker


class CategoricalRegressionHelper(CategoricalHelper):


    def __init__(self, dataHelper, model, description=""):
        """
        Constructor.

        Parameters
        ----------
        dataHelper : DataHelper
            DataHelper that has the data in a pandas.DataFrame.
        model : Model
            A regression model.
        description : string
            A description of the model.

        Returns
        -------
        None.
        """
        super().__init__(dataHelper, model, description)


    def CreateFeatureImportancePlot(self, titleSuffix=None, yFontScale=1.0, maximumFeatures=None, yAxisLabels=None, width=10, height=6):
        """
        Plots importance factors as a bar plot.

        Parameters
        ----------
        titleSuffix : string or None, optional
            If supplied, the string is prepended to the title.
        yFontScale : float
            Scale factor for the y axis labels.  If there are a lot of features, they tend to run together
            and may need to be shrunk.
        maximumFeatures : int or None, optional
            The maximum number of features to plot.  If None, all features are returned.
        yAxisLabels : list of strings, optional
            A list of names to replace the feature names as labels on the Y axis.  The most important
            feature (top of the chart) is first in the list.
        width : float, optional
            The width of the figure. The default is 10.
        height : float, optional
            The height of the figure. The default is 6.

        Returns
        -------
        None.
        """
        # Need the values in the reverse order (smallest to largest) for the bar plot to get the largest value on
        # the top (highest index position).
        importancesDataFrame = self.GetSortedImportance(ascending=True)

        # We cannot pass the maximuFeatures to GetSortedImportance because we are plotting in reverse.  That is, we want
        # the biggest values on the bottom of the DataFrame so we need to get the tail, not the head.
        if maximumFeatures is not None:
            importancesDataFrame = importancesDataFrame.tail(maximumFeatures)

        # We will start by just numbering the indices for the Y axis.  We will then rename and rescale them in a separate operation.
        indices = range(importancesDataFrame.shape[0])

        # Allow the features to be renamed.  List must be reversed because the plot is reversed.
        yLabels = importancesDataFrame.index
        if yAxisLabels is not None:
            yLabels = yAxisLabels.copy()
            yLabels.reverse()

        # Must be run before creating figure or plotting data.
        PlotHelper.Format(width=width, height=height)

        plt.barh(indices, importancesDataFrame["Importance"], color="cornflowerblue", align="center")
        plt.yticks(indices, yLabels, fontsize=12*PlotHelper.scale*yFontScale)
        AxesHelper.Label(plt.gca(), title="Feature Importances", xLabels="Relative Importance", titleSuffix=titleSuffix)

        plt.show()


    def GetSortedImportance(self, ascending=False, maximumFeatures=None):
        """
        Sorts the importance factors and returns them in a Pandas DataFrame.

        Parameters
        ----------
        ascending : bool
            Specifies if the values should be sorted as ascending or descending.
        maximumFeatures : int or None, optional
            The maximum number of features to return.  If None, all features are returned.

        Returns
        -------
        importances : pandas.DataFrame
            DataFrame of the sorted importance values.
        """
        index = None
        if type(self.dataHelper.xTrainingData) == pd.core.frame.DataFrame:
            index = self.dataHelper.xTrainingData.columns

        importances =  pd.DataFrame(self.model.feature_importances_,
                            columns=["Importance"],
                            index=index).sort_values(by="Importance", ascending=ascending)

        if maximumFeatures is not None:
            importances = importances.head(maximumFeatures)

        return importances


    def CreateConfusionMatrixPlot(self, dataSet="training", titleSuffix=None, axisLabels=None):
        """
        Plots the confusion matrix for the model output.

        Parameters
        ----------
        dataSet : string
            Which data set(s) to plot.
            training - Plots the results from the training data.
            testing  - Plots the results from the test data.
        titleSuffix : string or None, optional
            If supplied, the string is prepended to the title.
        axisLabels : array like of strings
            Labels to use on the predicted and actual axes.

        Returns
        -------
        confusionMatrix : ndarray of shape (n_classes, n_classes)
            The confusion matrix for the data.
        """
        confusionMatrix = self.GetConfusionMatrix(dataSet)

        PlotMaker.CreateConfusionMatrixPlot(confusionMatrix, dataSet.title()+" Data", titleSuffix, axisLabels)

        return confusionMatrix


    def GetConfusionMatrix(self, dataSet="training"):
        """
        Gets the confusion matrix for the model output.

        Parameters
        ----------
        dataSet : string
            Which data set(s) to plot.
            training - Plots the results from the training data.
            validation - Plots the result from the validation data.
            testing  - Plots the results from the testing data.
        scale : double
            Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot.

        Returns
        -------
        confusionMatrix : ndarray of shape (n_classes, n_classes)
            The confusion matrix for the data.
        """
        # Initialize to nothing.
        confusionMatrix = None

        # Get the confusion matrix for the correct data set.
        if dataSet == "training":
            if len(self.yTrainingPredicted) == 0:
                self.Predict()
            confusionMatrix = metrics.confusion_matrix(self.dataHelper.yTrainingData, self.yTrainingPredicted)

        elif dataSet == "validation":
            if len(self.yValidationPredicted) == 0:
                self.Predict()
            confusionMatrix = metrics.confusion_matrix(self.dataHelper.yValidationData, self.yValidationPredicted)

        elif dataSet == "testing":
            if len(self.yTestingPredicted) == 0:
                self.Predict()
            confusionMatrix = metrics.confusion_matrix(self.dataHelper.yTestingData, self.yTestingPredicted)

        else:
            raise Exception("Invalid data set specified.")

        return confusionMatrix