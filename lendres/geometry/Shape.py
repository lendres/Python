"""
Created on Augugst 12, 2022
@author: Lance A. Endres
"""
import numpy                                     as np


class Shape():


    def __init__(self):
        """
        Constructor.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.shapes = {}
        self.id     = id(self)


    def AddShape(self, shape):
        self.shapes[shape.id] = shape