"""
Created on June 27, 2022
@author: Lance A. Endres
"""
import numpy                                     as np

from   lendres.TensorFlowHelper                  import TensorFlowHelper


class TensorFlowMultiClassHelper(TensorFlowHelper):


    def __init__(self, tensorFlowDataHelper, model, description=""):
        """
        Constructor.

        Parameters
        ----------
        tensorFlowDataHelper : TensorFlowDataHelper
            TensorFlowDataHelper that has the data.
        model : Model
            A TensorFlow model.
        description : string
            A description of the model.

        Returns
        -------
        None.
        """
        super().__init__(tensorFlowDataHelper, model, description)


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
        yPredicted = self.model.predict(xData)
        yPredictedClass = []

        for i in yPredicted:
            # Get the index of the maximum value.
            yPredictedClass.append(np.argmax(i))

        return yPredictedClass