"""
Created on February 16, 2022
@author: Lance A. Endres
"""
import numpy                                     as np

class Search():
    """
    Searching algorithms.
    """


    @classmethod
    def BoundingBinarySearch(cls, item, points, returnedUnits="indices"):
        """
        Finds the bounding values for item in a list of points.

        The search algorithm is a binary search.

        Parameters
        ----------
        item : int or float
            Item to bound.
        points : int or float
            A list of points to search through.
        returnedUnits : string, optional
            Specifies the context of the returned values. The default is "indices".
                indices : Returns the indices of the "points" list.
                values : Returns the bounding values, that is values = points[indices].

        Returns
        -------
        list : int or float
            A list of length two that has either the indices or the values that bound the
            input "item."  If "item" is in "points," the list will contain two entries that
            are the same (the index/value).

            If "item" is not in the list, [np.NaN, np.NaN] is returned.
        """
        first = 0
        last  = len(points)-1

        # Check for out of range.
        if item < points[first] or item > points[last]:
            return [np.nan, np.nan]

        continueSearch = True

        while (continueSearch):

            # Find the midpoint index.
            midpoint = (first + last) // 2

            # Check to see if the value we are trying to bound is in the list.
            if points[midpoint] == item:
                first          = midpoint
                last           = midpoint
                continueSearch = False

            # This catches when the point is bounded.
            elif last - first < 2:
                continueSearch = False

            else:
                if item < points[midpoint]:
                    last = midpoint
                else:
                    first = midpoint

        # Return either the indices (position) of the array the bounded values are located at
        # or return the values that bound the input item.
        if returnedUnits == "indices":
            return [first, last]
        else:
            return [points[first], points[last]]


    @classmethod
    def FindIndicesByValues(cls, data, searchValue, maxCount=None):
        """
        Searches an array like object to find the indices of entries that match a specified value.

        Parameters
        ----------
        data : array like
            Data to search through.
        searchValue : string, numeric
            Value to match in the data.  Indices of entries that match this value are returned.
        maxCount : integer
            The maximum number of entries to returned.  If maxCount is reached before the end of the data,
            the function exits and returns the found values.  This allows returning 10 items even though 100
            are in the data, for example.
        Returns
        -------
        indices : list of integers
            The indices of all matched values.
        """
        # If no maximum was provided, we can return up to the total number.  We do it this way so we do not
        # have to check every time in the loop if maxCount is none and then check the value of maxCount if
        # it is not none.
        if maxCount == None:
            maxCount = len(data)

        foundCount  = 0
        indices     = []

        for i, value in enumerate(data):
            if value == searchValue:
                indices.append(i)
                foundCount += 1

                # If we reach the maximum number of entries, exit loop and return.
                if foundCount == maxCount:
                    break

        return indices