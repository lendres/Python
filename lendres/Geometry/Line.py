"""
Created on Augugst 12, 2022
@author: Lance A. Endres
"""
import numpy                                     as np
from   lendres.Geometry.Shape                    import Shape

class Line(Shape):


    def __init__(self, startPoint, endPoint):
        """
        Constructor.

        Parameters
        ----------
        x : float, optional
            X value
        y : float, optional
            Y value
        values : list of floats
            X and y values in a list.

        Returns
        -------
        None.
        """
        super().__init__()

        self.shapes[startPoint.id] = startPoint
        self.shapes[endPoint.id]   = endPoint

        startPoint.AddShape(self)
        endPoint.AddShape(self)