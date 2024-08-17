"""
Created on Augugst 12, 2022
@author: Lance A. Endres
"""
import numpy                                     as np

from   lendres.mathematics.Precision             import Precision
from   lendres.geometry.Shape                    import Shape


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


    @property
    def X(self):
        return self.values[0]


    @property
    def Y(self):
        return self.values[1]


    @property
    def Z(self):
        if len(self.values) < 3:
            raise Exception("Point is not contain three dimensions.")
        return self.values[2]


    def __add__(self, obj):
        # Adding two objects.
        size       = len(self.values)
        values     = [0] * size
        objectType = type(obj)

        if objectType == Point:
            for i in range(size):
                values[i] = self.values[i] + obj.values[i]

        elif objectType == list or objectType == np.ndarray:
            for i in range(size):
                values[i] = self.values[i] + obj[i]

        elif objectType == int or objectType == float:
            for i in range(size):
                values[i] = self.values[i] + obj

        else:
            raise Exception("Object type not found.  Object type:" + str(objectType))

        return Point(values)


    def __eq__(self, obj):
        # Adding two objects.
        size       = len(self.values)
        objectType = type(obj)

        if objectType == Point:
            for i in range(size):
                if not Precision.Equal(self.values[i], obj.values[i]):
                    return False
            return True

        elif objectType == list or objectType == np.ndarray:
            for i in range(size):
                if not Precision.Equal(self.values[i], obj[i]):
                    return False
            return True

        else:
            raise Exception("Object type not found.  Object type:" + str(objectType))


    def __ne__(self, obj):
        # Adding two objects.
        size       = len(self.values)
        objectType = type(obj)

        if objectType == Point:
            for i in range(size):
                if not Precision.Equal(self.values[i], obj.values[i]):
                    return True
            return False

        elif objectType == list or objectType == np.ndarray:
            for i in range(size):
                if not Precision.Equal(self.values[i], obj[i]):
                    return True
            return False

        else:
            raise Exception("Object type not found.  Object type:" + str(objectType))