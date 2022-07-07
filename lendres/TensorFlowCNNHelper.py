"""
Created on June 27, 2022
@author: Lance A. Endres
"""
import pandas                                    as pd
import numpy                                     as np

from   lendres.TensorFlowHelper                  import TensorFlowHelper

class TensorFlowCNNHelper(TensorFlowHelper):
    def __init__(self, imageHelper, model, description=""):
        """
        Constructor.

        Parameters
        ----------
        imageHelper : ImageHelper
            ImageHelper that has the input images.
        model : Model
            A TensorFlow model.
        description : string
            A description of the model.

        Returns
        -------
        None.
        """
        super().__init__(imageHelper, model, description)


    def DisplayModelEvaluation(self):
        """
        Displays the evalution of the model after it has run (summary of time and scores).

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        results = self.model.evaluate(self.dataHelper.xTestingData, self.dataHelper.yTestingEncoded)
        self.dataHelper.consoleHelper.Display(results)


    def Predict(self):
        """
        Predicts classification based on the maximum probabilities.
        Used for non-binary classification problems.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.yTrainingPredicted        = self.GetPredictedClassification(self.dataHelper.xTrainingData)
        self.yTestingPredicted         = self.GetPredictedClassification(self.dataHelper.xTestingData)

        if len(self.dataHelper.yValidationData) != 0:
            self.yValidationPredicted  = self.GetPredictedClassification(self.dataHelper.xValidationData)


    def GetPredictedClassification(self, xData):
        """
        Gets the predicted classification based on the maximum probabilities.
        Used for non-binary classification problems.

        Parameters
        ----------
        xData : array like
            Independent variables.

        Returns
        -------
        yPredictedData : array like
            Predictions of the dependent variable.
        """
        yPerdicted = self.model.predict(xData)
        yPerdictedClass = []

        for i in yPerdicted:
            yPerdictedClass.append(np.argmax(i))

        return yPerdictedClass