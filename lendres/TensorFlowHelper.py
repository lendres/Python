"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import pandas                                    as pd
import matplotlib.pyplot                         as plt

import seaborn                                   as sns
sns.set(color_codes=True)

from   sklearn                                   import metrics

import os

from   lendres.PlotHelper                        import PlotHelper
from   lendres.CategoricalHelper                 import CategoricalHelper

from   tensorflow.keras.models                   import load_model


class TensorFlowHelper(CategoricalHelper):
    # Class level variables.
    reportColumnLabels   = []
    modelResults         = {}
    numberOfOutputNodes  = 0


    def __init__(self, dataHelper, model, description=""):
        """
        Constructor.

        Parameters
        ----------
        dataHelper : DataHelper
            DataHelper that contains the data.
        model : TensorFlow Model or string
            A TensorFlow model or string that is the path to load the model from.
        description : string
            A description of the model.

        Returns
        -------
        None.
        """
        if type(model) == str:
            model = load_model(model)

        super().__init__(dataHelper, model, description)

        self.history     = None
        self.historyPath = ""

        TensorFlowHelper.SetNumberOfOutputNodes(model.layers[-1].output_shape[1])


    @classmethod
    def SetNumberOfOutputNodes(cls, numberOfNodes):
        """
        Uses the number of output nodes to format data storage used for reports.

        Parameters
        ----------
        numberOfNodes : int
            Number of output nodes.

        Returns
        -------
        None.
        """
        cls.numberOfOutputNodes = numberOfNodes

        # When there is one output node we get probabilitys for the negative score and positive score.
        numberOfNodeEntries = numberOfNodes
        if numberOfNodes == 1:
            numberOfNodeEntries = 2

        # Create an array of names.  If number of output nodes is 1, the array is:
        # ["Precision 0", "Precision 1", "Recall 0", "Recall 1", "F1 0", "F1 1", "Accuracy", "Error Rate"]

        # Create an entry for each metric and node number.
        cls.reportColumnLabels = []
        for name in ["Precision ", "Recall ", "F1 "]:
            for i in range(numberOfNodeEntries):
                cls.reportColumnLabels.append(name+str(i))

        # Add the final two categories.  These are overall and don't have individial entries for each node.
        cls.reportColumnLabels.append("Accuracy")
        cls.reportColumnLabels.append("Error Rate")


    def Fit(self, saveHistory=False, **kwargs):
        """
        Fits the model.

        Parameters
        ----------
        saveHistory : boolean
            If true, the history is appended to the previous, otherwise the history is ignored.
        **kwargs : keyword arguments
            These arguments are passed on to the model's fit function.

        Returns
        -------
        None.
        """
        if len(self.dataHelper.xTrainingData) == 0:
            raise Exception("The data has not been split.")

        if self.model == None:
            raise Exception("The model has not been created.")

        history = self.model.fit(
            self.dataHelper.xTrainingData,
            self.dataHelper.yTrainingEncoded,
            validation_data=(self.dataHelper.xValidationData, self.dataHelper.yValidationEncoded),
            **kwargs
        )

        if saveHistory:
            self.SetHistory(history, True)


    def UseHistorySaving(self, pathAndFileName):
        """
        Specifies that the history should be saved.  This will automatically try
        to load any existing history.

        Parameters
        ----------
        pathAndFileName : string
            Path to save and load the history from.

        Returns
        -------
        None.
        """
        self.historyPath = pathAndFileName
        self._LoadHistory(False)


    def SetHistory(self, tensorFlowHistory, appendHistory=True):
        """
        Sets the history.  Appends it to the current history if specified.

        Parameters
        ----------
        tensorFlowHistory : tensor flow history
            Tensor flow history to set.
        appendHistory : boolean
            If true, the history is appended to the current history.

        Returns
        -------
        None.
        """
        history = pd.DataFrame.from_dict(tensorFlowHistory.history)
        if appendHistory and self.history is not None:
            self.history = pd.concat([self.history, history], axis=0)
        else:
            self.history = history


    def SaveHistory(self):
        """
        Saves the hisory to the specified path.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.history.to_csv(self.historyPath, index=False)


    def _LoadHistory(self, raiseErrors=True):
        """
        Loads the hisory from the specified path.

        Parameters
        ----------
        path : string
            Path to load the history from.

        Returns
        -------
        None.
        """
        if os.path.exists(self.historyPath):
            self.dataHelper.consoleHelper.Print("Loading history from: " + self.historyPath)
            self.history = pd.read_csv(self.historyPath)
            self.dataHelper.consoleHelper.Print("History length: " + str(len(self.history)))
        elif raiseErrors:
            raise Exception("The specified path does not exist.\nPath: " + self.historyPath)


    def CreateTrainingAndValidationHistoryPlot(self, parameter):
        """
        Plots the confusion matrix for the model output.

        Parameters
        ----------
        parameter : string
            The parameter to plot.

        Returns
        -------
        figure : Matplotlib.Figure
        """
        # Must be called first.
        PlotHelper.FormatPlot()

        # Create x-values so that the first epoch is at 1 and not 0, the default plot start.
        xValues = range(1, len(self.history)+1)
        plt.plot(xValues, self.history[parameter])
        plt.plot(xValues, self.history["val_"+parameter])

        # Create titles and set legend.
        plt.gca().set(title="Model "+parameter.title(), xlabel="Epoch", ylabel=parameter.title())
        plt.legend(["Training", "Validation"], loc="best")

        figure = plt.gcf()

        plt.show()

        return figure


    def SaveClassificationReport(self):
        """
        Saves a classification report in a list for comparing with other models.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # Classification report for storage.  Get it as a dictionary so the values can be transfered to the DataFrame used to store the results.
        classificationReport = metrics.classification_report(self.dataHelper.yTestingData, self.yTestingPredicted, output_dict=True, zero_division=0)

        # Transfer the classification report dictionary to a DataFrame.
        scores = []

        # Create an entry for each metric and node number.
        for name in ["precision", "recall", "f1-score"]:
            for i in range(TensorFlowHelper.numberOfOutputNodes):
                scores.append(classificationReport[str(i)][name])

        # Accuracy and error rate.
        scores.append(classificationReport["accuracy"])
        scores.append(1-classificationReport["accuracy"])

        # Create a DataFrame from the scores and add them to the others.
        dataFrame = pd.DataFrame([scores], index=[self.GetName()], columns=TensorFlowHelper.reportColumnLabels)
        TensorFlowHelper.modelResults[self.GetName()] = dataFrame


    @classmethod
    def GetModelResults(cls):
        resultsDataFrame = pd.DataFrame(columns=cls.reportColumnLabels)

        for key in cls.modelResults:
            resultsDataFrame = pd.concat([resultsDataFrame, TensorFlowHelper.modelResults[key]], axis=0)

        return resultsDataFrame


    def DisplayClassificationReport(self):
        """
        Displays the classification report.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.dataHelper.consoleHelper.Print(metrics.classification_report(self.dataHelper.yTestingData, self.yTestingPredicted, zero_division=0))


    def GetDataSets(self, dataSet="testing"):
        """
        Gets the data sets based on the input argument.
        Parameters
        ----------
        dataSet : string
            Data set to select the image prediction from.  It can be either training, validation, or testing.
        Returns
        -------
        actualData : array like
            Data set of actual values.
        predictedData : array like
            Data set of predicted values.
        """
        actualData    = None
        predictedData = None
        if dataSet == "training":
            actualData    = self.dataHelper.yTrainingData
            predictedData = self.yTrainingPredicted
        elif dataSet == "validation":
            actualData    = self.dataHelper.yValidationData
            predictedData = self.yValidationPredicted
        elif dataSet == "testing":
            actualData    = self.dataHelper.yTestingData
            predictedData = self.yTestingPredicted
        else:
            raise Exception("The \"dataSet\" argument is invalid.")

        return actualData, predictedData


    def GetPredictions(self, dataSet="testing", criteria="wrong"):
        """
        Plot example image.
        Parameters
        ----------
        dataSet : string
            Data set to select the image prediction from.  It can be either training, validation, or testing.
        criteria : string
            Prediction criteria that is one of:
            correct : Returns the entries that were correctly predicted.
            wrong : Returns the entries that were incorrectly predicted.
        Returns
        -------
        None.
        """
        actualData, predictedData = self.GetDataSets(dataSet)

        wrongPredictions = None
        if criteria == "correct":
            wrongPredictions = actualData[actualData == predictedData]
        elif criteria == "wrong":
            wrongPredictions = actualData[actualData != predictedData]
        else:
            raise Exception("Incorrect value provided for the \"criteria\" argument")

        return wrongPredictions