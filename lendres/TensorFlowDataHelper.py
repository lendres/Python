"""
Created on July 29, 2022
@author: Lance A. Endres
"""
from   lendres.DataHelperBase                                   import DataHelper
from   lendres.TensorFlowDataHelperFunctions                    import TensorFlowDataHelperFunctions


class TensorFlowDataHelper(DataHelper):
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


    def EncodeDependentVariableForAI(self):
        """
        Converts the categorical columns ("category" data type) to encoded values.
        Prepares categorical columns for use in a model.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        TensorFlowDataHelperFunctions.EncodeDependentVariableForAI(self)


    def GetNumberOfUniqueCategories(self):
        """
        Gets the number of unique categories in the ouput.  This is the same as the number of classes in a classification problem.
        - Must be used after splitting the data.
        - Assumes all categories are represented in the dependent variable training data.  This is, if you split the data in such
          a way that one or more categories is not in the training data, it will not work.

        Parameters
        ----------
        None.

        Returns
        -------
        numberOfUniqueCategories : int
            Number of nodes in the ouput.  This is the same as the number of classes in a classification problem.
        """
        return TensorFlowDataHelperFunctions.GetNumberOfUniqueCategories(self)


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
        TensorFlowDataHelperFunctions.DisplayAIEncodingResults(self, numberOfEntries, randomEntries)


    def GetAIInputShape(self):
        """
        Gets the shape of the AI model input.

        Parameters
        ----------
        None.

        Returns
        -------
        : tuple
            Shape of input data.
        """
        return TensorFlowDataHelperFunctions.GetAIInputShape(self)