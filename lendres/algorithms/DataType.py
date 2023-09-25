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
        return all(isinstance(element, list) | isinstance(element, tuple) for element in inputList)


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
        return any(isinstance(element, list) | isinstance(element, tuple) for element in inputList)


    @classmethod
    def AreListsOfListsSameSize(cls, listOfLists1:list|tuple, listOfLists2:list|tuple):
        """
        Determines if a set of nested object are the same size.  To be the same size, each sub list/tuple/et cetera must be the same size

        Parameters
        ----------
        listOfLists1 : list|tuple
            First nested object.
        listOfLists2 : list|tuple
            Second nested object.


        Returns
        -------
        : bool
            True if every element and sub element are the same length, False otherwise.
        """
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


    @classmethod
    def GetLengthOfNestedObjects(cls, nestedObjects:list|tuple):
        """
        Counts the total number of objects is a list/tuple/et cetera.  Recursively counts the elements.
        Example:
            GetLengthOfNestedObjects([1, [2, 3]])
            result: 3

        Parameters
        ----------
        nestedObjects : list|tuple
            An iterable object.
        Returns
        -------
        : int
            Total number of elements in the object.
        """
        return cls._CountNestedObjects(0, nestedObjects)


    @classmethod
    def _CountNestedObjects(cls, count:int, obj:int|float|str|list|tuple):
        """
        Drills down iterable objects counting each element as it goes.
        Counts an element if it is not iterable (e.g. an int) or iteratotes over each element if it is iterable.

        Parameters
        ----------
        count : int
            The current count of individual objects.
        obj : int|list|tuple
            Current object of interest.

        Returns
        -------
        count : int
            The current count of found individual elements.
        """
        match obj:
            # For single elements we add to the count.
            case int() | float() | str():
                return count + 1

            # For iterable objects we loop over them and call ourself to continue to interigation.
            case list() | tuple():
                for item in obj:
                    count = cls._CountNestedObjects(count, item)
                return count

            # Catch any data types that have not been acounted for and raise an error.
            case _:
                raise Exception("Unknown object type.")