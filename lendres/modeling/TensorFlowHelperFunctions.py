"""
Created on July 29, 2022
@author: Lance A. Endres
"""
import pandas                                                        as pd
import numpy                                                         as np
import tensorflow                                                    as tf
import random

from   lendres.io.ConsoleHelper                                      import ConsoleHelper

class TensorFlowDataHelperFunctions():
    """
    Class for common code for manipulating data for use in an artificial intelligence setting.

    The DataHelper and ImageHelper work very differently, but both need to be used for TensorFlow.  To support this
    the common code is here and the TensorFlowDataHelper and ImageHelper reference these functions to do the
    work.

    It was decided to keep all TensorFlow materials out of the DataHelper class so it could be used without having
    to install the TensorFlow library.  The only other solution was to create very odd diamond shape inheritance
    scheme.

    This class adds a separate encoding data set for the dependent variables.  In artificial intelligence, the dependent
    data has to be encoded as an array of 0s and 1s.  For binary classification, this means we only need one column.  For
    multiple class classification we need an array for each data sample that indicates which class the sample is in.


    General Notes
        - Split the data before preprocessing the data.
            - The class is set up so the original data is preserved in the dataHelper.data variable and
              the processed data is in the split variables (xTrainingData, xValidationData, xTestingdata).
    """

    @classmethod
    def EncodeDependentVariableForAI(cls, dataHelper):
        """
        Converts the categorical columns ("category" data type) to encoded values.
        Prepares categorical columns for use in a model.

        Parameters
        ----------
        dataHelper : DataHelper or ImageHelper
            The data storage container.

        Returns
        -------
        None.
        """
        numberOfUniqueCategories = cls.GetNumberOfUniqueCategories()

        # For binary classification, we don't want to change the data.  We already have 1 column of 0/1s.
        # For multiclass classification we need an array of 0s and 1s, one for each potential class.
        processingFunction = None
        if numberOfUniqueCategories == 2:
            processingFunction = lambda data : data
        elif numberOfUniqueCategories > 2:
            processingFunction = lambda data : tf.keras.utils.to_categorical(data)
        else:
            raise Exception("Invalid number or entries found.")

        dataHelper.yTrainingEncoded       = processingFunction(dataHelper.yTrainingData)
        if len(dataHelper.yValidationData) != 0:
            dataHelper.yValidationEncoded = processingFunction(dataHelper.yValidationData)
        dataHelper.yTestingEncoded        = processingFunction(dataHelper.yTestingData)


    @classmethod
    def GetNumberOfUniqueCategories(cls, dataHelper):
        """
        Gets the number of unique categories in the ouput.  This is the same as the number of classes in a classification problem.
        - Must be used after splitting the data.
        - Assumes all categories are represented in the dependent variable training data.  This is, if you split the data in such
          a way that one or more categories is not in the training data, it will not work.

        Parameters
        ----------
        dataHelper : DataHelper or ImageHelper
            The data storage container.

        Returns
        -------
        numberOfUniqueCategories : int
            Number of nodes in the ouput.  This is the same as the number of classes in a classification problem.
        """
        # The length function is used because both numpy arrays and pandas.Series have unique functions, but
        # numpy arrays do not have an nunique function.  This way lets us operate on both without having to check
        # the data type.
        numberOfUniqueCategories = 0
        yDataType                = type(dataHelper.yTrainingData)
        if yDataType == np.ndarray:
            numberOfUniqueCategories = len(np.unique(dataHelper.yTrainingData))
        elif yDataType == pd.core.series.Series:
            numberOfUniqueCategories = dataHelper.yTrainingData.nunique()
        else:
            raise Exception("Data type is unknown.")

        return numberOfUniqueCategories


    @classmethod
    def DisplayAIEncodingResults(cls, dataHelper, numberOfEntries, randomEntries=False):
        """
        Prints a summary of the encoding processes.

        Parameters
        ----------
        dataHelper : DataHelper or ImageHelper
            The data storage container.
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
            numberOfSamples = len(dataHelper.yTrainingEncoded)
            indices = random.sample(range(0, numberOfSamples), numberOfEntries)
        else:
            indices = list(range(numberOfEntries))

        dataHelper.consoleHelper.PrintTitle("Dependent Variable Numerical Labels", verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        yNumbers = dataHelper.yTrainingData.iloc[indices]
        dataHelper.consoleHelper.Display(pd.DataFrame(yNumbers), verboseLevel=ConsoleHelper.VERBOSEREQUESTED)

        dataHelper.consoleHelper.PrintNewLine(verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        dataHelper.consoleHelper.PrintTitle("Dependent Variable Text Labels", verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        dataHelper.consoleHelper.PrintTitle("*** NEEDS FIXING ***", verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        #labels = [dataHelper.labelCategories[i] for i in yNumbers]
        #dataHelper.consoleHelper.Display(pd.DataFrame(labels, columns=["Labels"], index=yNumbers.index), ConsoleHelper.VERBOSEREQUESTED)

        dataHelper.consoleHelper.PrintNewLine(verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        dataHelper.consoleHelper.PrintTitle("Dependent Variable Encoded Labels", verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        dataHelper.consoleHelper.Display(pd.DataFrame(dataHelper.yTrainingEncoded[indices], index=yNumbers.index), verboseLevel=ConsoleHelper.VERBOSEREQUESTED)


    @classmethod
    def GetAIInputShape(cls, dataHelper):
        """
        Gets the shape of the AI model input.

        Parameters
        ----------
        dataHelper : DataHelper or ImageHelper
            The data storage container.

        Returns
        -------
        tuple
            Shape of input data.
        """
        return dataHelper.xTrainingData[0].shape