"""
Created on Augugst 12, 2022
@author: Lance A. Endres
"""
import numpy                                     as np

from   lendres.Mathematics.Precision             import Precision
from   lendres.Geometry.Shape                    import Shape


class Point(Shape):


    def __init__(self, values=None):
        """
        Constructor.

        Parameters
        ----------
        values : list of floats
            X and y values in a list.

        Returns
        -------
        None.
        """
        super().__init__()

        if values is not None:
            self.values = np.array(values)


    def __add__(self, obj):
        # Adding two objects.
        size   = len(self.values)
        values = [0] * size

        objectType = type(obj)

        if objectType == Point:
            for i in range(size):
                values[i] = self.values[i] + obj.values[i]

        if objectType == int or objectType == float:
            for i in range(size):
                values[i] = self.values[i] + obj

        return Point(values)


    def EqualsPoint(self, point):
        size = len(self.values)

        if size != len(point.values):
            return False

        for i in range(size):
            equal = Precision.Equal(self.values[i], point.values[i])
            if not equal:
                return False
        return True


    def EqualsValues(self, values):
        size = len(self.values)

        if size != len(values):
            return False

        for i in range(size):
            equal = Precision.Equal(self.values[i], values[i])
            if not equal:
                return False
        return True