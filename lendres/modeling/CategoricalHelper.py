"""
Created on July 10, 2022
@author: Lance A. Endres
"""
from   sklearn                                                       import metrics

from   lendres.modeling.ModelHelper                                  import ModelHelper
from   lendres.plotting.PlotMaker                                    import PlotMaker


class CategoricalHelper(ModelHelper):


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


    def CreateConfusionMatrixPlot(self, dataSet="training", titleSuffix=None, axisLabels=None):
        """
        Plots the confusion matrix for the model output.

        Parameters
        ----------
        dataSet : string
            Which data set(s) to plot.
            training   - Plots the results from the training data.
            validation - Plots the result from the validation data.
            testing    - Plots the results from the test data.
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

        PlotMaker.CreateConfusionMatrixPlot(confusionMatrix, dataSet.title()+" Data", titleSuffix=titleSuffix, axesLabels=axisLabels)

        return confusionMatrix


    def GetConfusionMatrix(self, dataSet="training"):
        """
        Gets the confusion matrix for the model output.

        Parameters
        ----------
        dataSet : string
            Which data set(s) to plot.
            training   - Plots the results from the training data.
            validation - Plots the result from the validation data.
            testing    - Plots the results from the testing data.
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