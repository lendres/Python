"""
Created on Augugst 12, 2022
@author: Lance A. Endres
"""
import numpy                                     as np
from   lendres.Geometry.RotationDirection        import RotationDirection
from   lendres.Geometry.Shape                    import Shape
from   lendres.LinearAlgebra                     import AngleIn360Degrees
from   lendres.LinearAlgebra                     import DiscritizeArc


class Arc(Shape):
    """
    A constant radius arc.
    Defined as counter-clockwise.
    """

    def __init__(self, centerPoint, startPoint, endPoint, rotationDirection=RotationDirection.Positive):
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

        # The center point is a control point, so it is kept separate.
        self.center  = centerPoint

        self.shapes["start"]   = startPoint
        self.shapes["end"]     = endPoint

        self.rotationDirection = rotationDirection

        # The center is a control point, so it is not added.
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
        # Get the starting and ending angles.
        startAngle = AngleIn360Degrees(startPoint=self.center.values, endPoint=self.shapes["start"].values)
        endAngle   = AngleIn360Degrees(startPoint=self.center.values, endPoint=self.shapes["end"].values)

        # The discritize function requires a positive direction.  So we reverse the angles if the the arc is negative.
        if self.rotationDirection == RotationDirection.Negative:
            temp       = endAngle
            endAngle   = startAngle
            startAngle = temp

        points = DiscritizeArc(self.center.values, self.GetRadius(), startAngle, endAngle, numberOfPoints)

        # If the arc goes in the negative direction we have to reverse the points so they come back in the expected order.
        if self.rotationDirection == RotationDirection.Negative:
            points = np.flip(points, axis=0)

        return points