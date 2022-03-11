# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 05:52:33 2022

@author: Lance
"""
import numpy as np

def BoundingBinarySearch(item, points, returnedUnits="indices"):
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
        elif item > points[midpoint-1] and item < points[midpoint]:
            first          = midpoint-1
            last           = midpoint
            continueSearch = False

        else:
            if item < points[midpoint]:
                last = midpoint - 1
            else:
                first = midpoint + 1

    # Return either the indices (position) of the array the bounded values are located at
    # or return the values that bound the input item.
    if returnedUnits == "indices":
        return [first, last]
    else:
        return [points[first], points[last]]