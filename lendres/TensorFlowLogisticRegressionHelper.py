"""
Created on June 27, 2022
@author: Lance A. Endres
"""

from   lendres.TensorFlowHelper                  import TensorFlowHelper

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


    def Fit(self, **kwargs):
        """
        Fits the model.

        Parameters
        ----------
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

        self.history = self.model.fit(
            self.dataHelper.xTrainingData,
            self.dataHelper.yTrainingData,
            **kwargs
        )


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


    def DisplayModelEvaluation(self):
        results = self.model.evaluate(self.dataHelper.xTestingData, self.dataHelper.yTestingData)
        self.dataHelper.consoleHelper.Display(results)