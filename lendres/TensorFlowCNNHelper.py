"""
Created on June 27, 2022
@author: Lance A. Endres
"""
import numpy                                     as np

from   lendres.ImageHelper                       import ImageHelper
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
        actualData, predictedData = self.GetDataSets(dataSet)

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

        xData, yData = self.dataHelper.GetDataSet(dataSet)

        ImageHelper.PlotImage(xData[location], title=actualName, size=size)