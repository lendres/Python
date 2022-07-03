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

        history = self.model.fit(
            self.dataHelper.xTrainingData,
            self.dataHelper.yTrainingEncoded,
            validation_data=(self.dataHelper.xValidationData, self.dataHelper.yValidationEncoded),
            **kwargs
        )

        self.history = pd.DataFrame.from_dict(history.history)


    def SaveHistory(self, path):
        self.history.to_csv(path, index=False)


    def LoadHistory(self, inputFile):
        self.history = pd.read_csv(inputFile)


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


    def ComparePredictedToActual(self, dataSet="testing", index=None, location=None, random=False, size=6):
        """
        Plot example image.

        Parameters
        ----------
        dataSet : string
            Data set to select the image prediction from.  It can be either training, validation, or testing.
        index : index value
            Index value of image to plot.
        location : int
            Location of image to plot (selected using iloc).
        random : boolean
            If true, a random image is selected.
        size : float
            Size (width and height) of figure.

        Returns
        -------
        None.
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

        if index != None:
            location  = actualData.index.get_loc(index)

        if location != None:
            index     = actualData.index[location]

        # Generating random indices from the data and plotting the images.
        if random:
            location  = np.random.randint(0, len(actualData))
            index     = actualData.index[location]

        # Predicted data doesn't have an index, we need to use its location.
        predictedName = self.dataHelper.labelCategories[predictedData[location]]
        actualName    = self.dataHelper.labels["Names"].loc[index]

        self.dataHelper.consoleHelper.PrintBold("Predicted: "+predictedName)
        self.dataHelper.consoleHelper.PrintBold("Actual:    "+actualName)
        self.dataHelper.consoleHelper.PrintNewLine()

        self.dataHelper.PlotImage(index, size=size)