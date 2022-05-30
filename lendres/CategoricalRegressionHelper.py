"""
Created on January 19, 2022
@author: Lance
"""
import pandas                   as pd
import numpy                    as np

import matplotlib.pyplot        as plt
import seaborn                  as sns

from sklearn                    import metrics

from lendres.ConsoleHelper      import ConsoleHelper
from lendres.ModelHelper        import ModelHelper
from lendres.PlotHelper         import PlotHelper
from lendres.PlotMaker          import PlotMaker

class CategoricalRegressionHelper(ModelHelper):

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


    def CreateFeatureImportancePlot(self, titlePrefix=None, yFontScale=1.0):
        """
        Plots importance factors as a bar plot.

        Parameters
        ----------
        titlePrefix : string or None, optional
            If supplied, the string is prepended to the title.
        yFontScale : float
            Scale factor for the y axis labels.  If there are a lot of features, they tend to run together
            and may need to be shrunk.

        Returns
        -------
        None.
        """
        # Need the values in the reverse order (smallest to largest) for the bar plot to get the largest value on
        # the top (highest index position).
        importancesDataFrame = self.GetSortedImportance(ascending=True)
        indices              = range(importancesDataFrame.shape[0])

        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        plt.barh(indices, importancesDataFrame["Importance"], color="cornflowerblue", align="center")
        plt.yticks(indices, importancesDataFrame.index, fontsize=12*PlotHelper.scale*yFontScale)
        PlotHelper.Label(plt.gca(), title="Feature Importances", xLabel="Relative Importance", titlePrefix=titlePrefix)

        plt.show()


    def GetSortedImportance(self, ascending=False):
        """
        Sorts the importance factors and returns them in a Pandas DataFrame.

        Parameters
        ----------
        ascending : bool
            Specifies if the values should be sorted as ascending or descending.

        Returns
        -------
        : pandas.DataFrame
            DataFrame of the sorted importance values.
        """
        return pd.DataFrame(self.model.feature_importances_,
                            columns=["Importance"],
                            index=self.dataHelper.xTrainingData.columns).sort_values(by="Importance", ascending=ascending)


    def CreateConfusionMatrixPlot(self, dataSet="training", titlePrefix=None):
        """
        Plots the confusion matrix for the model output.

        Parameters
        ----------
        dataSet : string
            Which data set(s) to plot.
            training - Plots the results from the training data.
            testing  - Plots the results from the test data.
        titlePrefix : string or None, optional
            If supplied, the string is prepended to the title.

        Returns
        -------
        confusionMatrix : ndarray of shape (n_classes, n_classes)
            The confusion matrix for the data.
        """
        confusionMatrix = self.GetConfusionMatrix(dataSet)

        PlotMaker.CreateConfusionMatrixPlot(confusionMatrix, dataSet.title()+" Data", titlePrefix)

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


    def DisplayModelPerformanceScores(self, final=False):
        """
        Displays the model performance scores based on the settings in the ConsuleHelper.

        Returns
        -------
        None.
        """
        scores = self.GetModelPerformanceScores(final)
        self.dataHelper.consoleHelper.PrintTitle("Performance Scores", ConsoleHelper.VERBOSEREQUESTED)
        self.dataHelper.consoleHelper.Display(scores, ConsoleHelper.VERBOSEREQUESTED)


    def GetModelPerformanceScores(self, final=False):
        """
        Calculate performance metrics.  Threshold for a positive result can be specified.

        Parameters
        ----------
        threshold : float
            Threshold for classifying the observation success.

        Returns
        -------
        dataFrame : DataFrame
            DataFrame that contains various performance scores for the training and test data.
        """
        # Make sure the model has been initiated and of the correct type.
        if self.model == None:
            raise Exception("The regression model has not be initiated.")

        if len(self.yTrainingPredicted) == 0:
            raise Exception("The predicted values have not been calculated.")

        # Calculate scores.
        # TRAINING.
        # Accuracy.
        accuracyScores   = [metrics.accuracy_score(self.dataHelper.yTrainingData, self.yTrainingPredicted)]
        # Recall.
        recallScores     = [metrics.recall_score(self.dataHelper.yTrainingData, self.yTrainingPredicted)]
        # Precision.
        precisionScores  = [metrics.precision_score(self.dataHelper.yTrainingData, self.yTrainingPredicted, zero_division=0)]
        # F1.
        f1Scores         = [metrics.f1_score(self.dataHelper.yTrainingData, self.yTrainingPredicted)]
        # Index.
        index            = ["Training"]

        # VALIDATION.
        if len(self.dataHelper.yValidationData) != 0:
           # Accuracy.
            accuracyScores.append(metrics.accuracy_score(self.dataHelper.yValidationData, self.yValidationPredicted))
            # Recall.
            recallScores.append(metrics.recall_score(self.dataHelper.yValidationData, self.yValidationPredicted))
            # Precision.
            precisionScores.append(metrics.precision_score(self.dataHelper.yValidationData, self.yValidationPredicted, zero_division=0))
            # F1.
            f1Scores.append(metrics.f1_score(self.dataHelper.yValidationData, self.yValidationPredicted))
            # Index.
            index.append("Validation")

        if final:
            # TESTING.
            # Accuracy.
            accuracyScores.append(metrics.accuracy_score(self.dataHelper.yTestingData, self.yTestingPredicted))
            # Recall.
            recallScores.append(metrics.recall_score(self.dataHelper.yTestingData, self.yTestingPredicted))
            # Precision.
            precisionScores.append(metrics.precision_score(self.dataHelper.yTestingData, self.yTestingPredicted, zero_division=0))
            # F1.
            f1Scores.append(metrics.f1_score(self.dataHelper.yTestingData, self.yTestingPredicted))
            # Index.
            index.append("Testing")

        # Create a DataFrame for returning the values.
        dataFrame = pd.DataFrame({"Accuracy"  : accuracyScores,
                                  "Recall"    : recallScores,
                                  "Precision" : precisionScores,
                                  "F1"        : f1Scores},
                                 index=index)

        return dataFrame