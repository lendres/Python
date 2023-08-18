"""
Created on June 27, 2022
@author: Lance A. Endres
"""

from   lendres.TensorFlowHelper                                 import TensorFlowHelper


class TensorFlowLogisticRegressionHelper(TensorFlowHelper):
    # Class level variables.

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
        super().__init__(dataHelper, model, 1, description)


    def Predict(self, threshold=0.5):
        """
        Gets the predicted values based on a threshold.
        Used for logistic regression modeling.

        Parameters
        ----------
        threshold : float in range of 0-1
            The threshold used to determine if a value is predicted as true.

        Returns
        -------
        None.
        """
        # Predict on the training and testing data.
        self.yTrainingPredicted        = self.GetPredictedValues(self.dataHelper.xTrainingData, threshold)
        self.yTestingPredicted         = self.GetPredictedValues(self.dataHelper.xTestingData, threshold)

        if len(self.dataHelper.yValidationData) != 0:
            self.yValidationPredicted  = self.GetPredictedValues(self.dataHelper.xValidationData, threshold)


    def GetPredictedValues(self, xData, threshold=0.5):
        """
        Gets the predicted values based on a threshold.
        Used for logistic regression modeling.

        Parameters
        ----------
        threshold : float in range of 0-1
            The threshold used to determine if a value is predicted as true.

        Returns
        -------
        yPredictedData : array like
            Predictions of the dependent variable.
        """
        yPerdictedData = self.model.predict(xData)
        yPerdictedData = (yPerdictedData > threshold)
        return yPerdictedData