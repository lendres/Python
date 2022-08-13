"""
Created on Augugst 12, 2022
@author: Lance A. Endres
"""
import numpy                                     as np
from   lendres.Geometry.Shape                   import Shape
from   lendres.LinearAlgebra                     import AngleIn360Degrees
from   lendres.LinearAlgebra                     import DiscritizeArc


class Arc(Shape):
    """
    A constant radius arc.
    Defined as counter-clockwise.
    """

    def __init__(self, centerPoint, startPoint, endPoint, counterClockwise=True):
        """
        Constructor.

        Parameters
        ----------
        centerPoint : Point
            Center point of arc.
        startPoint : Point
            Y value
        startPoint : Point
            X and y values in a list.

        Returns
        -------
        None.
        """
        super().__init__()

        self.center = centerPoint

        if counterClockwise:
            self.shapes["start"]  = startPoint
            self.shapes["end"]    = endPoint
        else:
            self.shapes["start"]  = endPoint
            self.shapes["end"]    = startPoint

        #centerPoint.AddShape(self)
        startPoint.AddShape(self)
        endPoint.AddShape(self)


    def GetRadius(self):
        return np.linalg.norm(self.shapes["end"].values - self.center.values)


    def GetDiameter(self):
        return 2 * self.GetRadius()


    def GetStartAngle(self):
        return AngleIn360Degrees(startPoint=self.center, endPoint=self.shapes["start"])


    def GetEndAngle(self):
        return AngleIn360Degrees(startPoint=self.center, endPoint=self.shapes["end"])


    def Discritize(self, numberOfPoints=100):
        startAngle = AngleIn360Degrees(startPoint=self.center.values, endPoint=self.shapes["start"].values)
        endAngle   = AngleIn360Degrees(startPoint=self.center.values, endPoint=self.shapes["end"].values)
        return DiscritizeArc(self.center.values, self.GetRadius(), startAngle, endAngle, numberOfPoints)