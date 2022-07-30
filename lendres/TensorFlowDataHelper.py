"""
Created on July 29, 2022
@author: Lance A. Endres
"""
import pandas                                    as pd
import numpy                                     as np
import tensorflow                                as tf
import random

import lendres
from   lendres.DataHelperBase                    import DataHelperBase


class TensorFlowDataHelper(DataHelperBase):
    """
    Class for storing and manipulating data for use in an artificial intelligence setting.

    This class adds a separate encoding data set for the dependent variables.  In artificial intelligence, the dependent
    data has to be encoded as an array of 0s and 1s.  For binary classification, this means we only need one column.  For
    multiple class classification we need an array for each data sample that indicates which class the sample is in.


    General Notes
        - Split the data before preprocessing the data.
            - The class is set up so the original data is preserved in the self.data variable and
              the processed data is in the split variables (xTrainingData, xValidationData, xTestingdata).
    """

    def __init__(self, consoleHelper=None):
        """
        Constructor.

        Parameters
        ----------
        consoleHelper : ConsoleHelper
            Class the prints messages.

        Returns
        -------
        None.
        """
        super().__init__(consoleHelper)

        # Initialize the variable.  Helpful to know if something goes wrong.
        self.yTrainingEncoded        = []
        self.yValidationEncoded      = []
        self.yTestingEncoded         = []


    def CopyFrom(self, dataHelper):
        """
        Copies the data of the DataHelper supplied as input to this DataHelper.

        Parameters
        ----------
        dataHelper : DataHelperBase subclass.
            DataHelper to copy data from.

        Returns
        -------
        None.
        """
        super().CopyFrom(dataHelper)

        self.yTrainingEncoded        = dataHelper.yTrainingEncoded.copy()
        self.yValidationEncoded      = dataHelper.yValidationEncoded.copy()
        self.yTestingEncoded         = dataHelper.yTestingEncoded.copy()


    def EncodeDependentVariableForAI(self):
        """
        Converts the categorical columns ("category" data type) to encoded values.
        Prepares categorical columns for use in a model.

        Parameters
        ----------
        columns : list of strings
            The names of the columns to encode.
        dropFirst : bool
            If true, the first category is dropped for the encoding.

        Returns
        -------
        None.
        """
        # The length function is used because both numpy arrays and pandas.Series have unique functions, but
        # numpy arrays do not have an nunique function.  This way lets us operate on both without having to check
        # the data type.
        numberOfUniqueCategories = 0
        yDataType                = type(self.yTrainingData)
        if yDataType == np.ndarray:
            numberOfUniqueCategories = len(np.unique(self.yTrainingData))
        elif yDataType == pd.core.series.Series:
            numberOfUniqueCategories = self.yTrainingData.nunique()
        else:
            raise Exception("Data type is unknown.")

        # For binary classification, we don't want to change the data.  We already have 1 column of 0/1s.
        # For multiclass classification we need an array of 0s and 1s, one for each potential class.
        processingFunction = None
        if numberOfUniqueCategories == 2:
            processingFunction = lambda data : data
        elif numberOfUniqueCategories > 2:
            processingFunction = lambda data : tf.keras.utils.to_categorical(data)
        else:
            raise Exception("Invalid number or entries found.")

        self.yTrainingEncoded       = processingFunction(self.yTrainingData)
        if len(self.yValidationData) != 0:
            self.yValidationEncoded = processingFunction(self.yValidationData)
        self.yTestingEncoded        = processingFunction(self.yTestingData)


    def DisplayAIEncodingResults(self, numberOfEntries, randomEntries=False):
        """
        Prints a summary of the encoding processes.

        Parameters
        ----------
        numberOfEntries : int
            The number of entries to display.
        randomEntries : bool
            If true, random entries are chosen, otherwise, the first few entries are displayed.

        Returns
        -------
        None.
        """
        indices = []
        if randomEntries:
            numberOfImages = len(self.yTrainingEncoded)
            indices = random.sample(range(0, numberOfImages), numberOfEntries)
        else:
            indices = list(range(numberOfEntries))

        self.consoleHelper.PrintTitle("Dependent Variable Numerical Labels")
        yNumbers = self.yTrainingData.iloc[indices]
        self.consoleHelper.Display(pd.DataFrame(yNumbers))

        self.consoleHelper.PrintNewLine()
        self.consoleHelper.PrintTitle("Dependent Variable Text Labels")
        labels = [self.labelCategories[i] for i in yNumbers]
        self.consoleHelper.Display(pd.DataFrame(labels, columns=["Labels"], index=yNumbers.index))

        self.consoleHelper.PrintNewLine()
        self.consoleHelper.PrintTitle("Dependent Variable Encoded Labels")
        self.consoleHelper.Display(pd.DataFrame(self.yTrainingEncoded[indices], index=yNumbers.index))