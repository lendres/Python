"""
Created on July 24, 2023
@author: Lance A. Endres
"""

class DataType():
    """
    A class for checking and determining data types.  These are advanced checks, for example,
    checking elements of a list.
    """

    @classmethod
    def IsListOfLists(cls, inputList):
        """
        Determine is the input contains elements that are all lists.

        Parameters
        ----------
        inputList : array like
            Input list.

        Returns
        -------
        : boolean
            Returns true if ALL the elements of the list are lists.
        """
        return all(isinstance(element, list) for element in inputList)


    @classmethod
    def ContainsAtLeastOneList(cls, inputList):
        """
        Determine is the input contains elements that are all lists.

        Parameters
        ----------
        inputList : array like
            Input list.

        Returns
        -------
        : boolean
            Returns true if ANY the elements of the list are lists.
        """
        return any(isinstance(element, list) for element in inputList)


    @classmethod
    def AreListsOfListsSameSize(cls, listOfLists1, listOfLists2):
        if not (cls.IsListOfLists(listOfLists1) and cls.IsListOfLists(listOfLists2)):
            raise Exception("At least one of the inputs is not a list of lists.")

        sizes1 = cls.GetSizesOfListOfLists(listOfLists1)
        sizes2 = cls.GetSizesOfListOfLists(listOfLists2)

        return sizes1 == sizes2


    @classmethod
    def GetSizesOfListOfLists(cls, listOfLists):
        return [len(element) for element in listOfLists]


    @classmethod
    def CreateListOfLists(cls, sizes, initializationValue=0):
        if not cls.IsListOfLists(sizes):
            raise Exception("The input sizes is not a list of lists.")

        result = [[initializationValue for size in sizeList] for sizeList in sizes]
        return result