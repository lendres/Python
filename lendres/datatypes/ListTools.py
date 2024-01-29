"""
Created on July 24, 2023
@author: Lance A. Endres
"""
from   collections.abc                                               import Iterable


class ListTools():
    """
    A class for checking and determining data types.  These are advanced checks, for example,
    checking elements of a list.
    """

    @classmethod
    def IsListOfLists(cls, inputList:list|tuple):
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
    def ContainsAtLeastOneList(cls, inputList:list|tuple):
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
    def GetSizesOfListOfLists(cls, listOfLists:list) -> list:
        """
        Gets the sizes of nested lists and returns them as a list.

        Parameters
        ----------
        listOfLists : list of lists
            A list that contains other lists.

        Returns
        -------
        list
            A list that contains the length (size) of each nest list supplied as input.
        """
        return [len(element) for element in listOfLists]


    @classmethod
    def CreateListOfLists(cls, sizes:list, initializationValue=0) ->list:
        """
        Creates a list of lists and populates them with the initialization value.

        Parameters
        ----------
        sizes : list of lists
            The length (size) of each list to create.
        initializationValue : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        list
            A list with nested lists populated with the initialization value.
        """
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


    @classmethod
    def Flatten(cls, nestedObjects:list|tuple) -> list:
        """
        Takes a list that contains nested lists and turns it into a single list.  The list may contain
        any combination of single values and lists.

        Example:
            In:  [1, [2, [3, 4]]]
            Out: [1, 2, 3, 4]

        Parameters
        ----------
        nestedObjects : list|tuple
            A list or tuple that may contain other lists/tuples.

        Returns
        -------
        list
            All the elements of the input as a single, flat list.
        """
        output = []

        for obj in nestedObjects:
            if isinstance(obj, Iterable) and not isinstance(obj, str):
                flattened = cls.Flatten(obj)
                for item in flattened:
                    output.append(item)
            else:
                output.append(obj)

        return output