"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import pandas                                    as pd
import matplotlib.pyplot                         as plt

import seaborn                                   as sns
sns.set(color_codes=True)

from   sklearn                                   import metrics

from   lendres.PlotHelper                        import PlotHelper
from   lendres.ModelHelper                       import ModelHelper
from   lendres.PlotMaker                         import PlotMaker

class TensorFlowHelper(ModelHelper):
    # Class level variables.
    reportColumnLabels = ["Precision 0", "Precision 1", "Recall 0", "Recall 1", "F1 0", "F1 1", "Accuracy", "Error Rate"]
    modelResults       = pd.DataFrame(columns=reportColumnLabels)


    def __init__(self, dataHelper, model, description=""):
        """
        Constructor.

        Parameters
        ----------
        dataHelper : DataHelper
            DataHelper that has the data in a pandas.DataFrame.
        model : Model
            A TensorFlow model.
        description : string
            A description of the model.

        Returns
        -------
        None.
        """
        self.history = None

        super().__init__(dataHelper, model, description)


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

        plt.plot(self.history.history[parameter])
        plt.plot(self.history.history["val_"+parameter])

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
        dataFrame = pd.DataFrame(
            [[
            classificationReport["0"]["precision"],
            classificationReport["1"]["precision"],
            classificationReport["0"]["recall"],
            classificationReport["1"]["recall"],
            classificationReport["0"]["f1-score"],
            classificationReport["1"]["f1-score"],
            classificationReport["accuracy"],
            1-classificationReport["accuracy"],
            ]],
            index=[tensorFlowHelper.GetName()],
            columns=TensorFlowHelper.reportColumnLabels
        )

        # Tell Python we are accessing a global variable.  Absolutely terrible practice by software standards, but that's
        # the Python way and it's nice and easy.
        TensorFlowHelper.modelResults = pd.concat([TensorFlowHelper.modelResults, dataFrame], axis=0)


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


    def CreateConfusionMatrixPlot(self, titlePrefix=None, axisLabels=None):
        """
        Plots the confusion matrix for the model output.

        Parameters
        ----------
        titlePrefix : string or None, optional
            If supplied, the string is prepended to the title.
        axisLabels : array like of strings
            Labels to use on the predicted and actual axes.

        Returns
        -------
        figure : Figure
            The newly created figure.
        """
        confusionMatrix = metrics.confusion_matrix(self.dataHelper.yTestingData, self.yTestingPredicted)
        return PlotMaker.CreateConfusionMatrixPlot(confusionMatrix, "Confusion Matrix", titlePrefix=titlePrefix, axisLabels=axisLabels)