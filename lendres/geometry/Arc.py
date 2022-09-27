"""
Created on Augugst 12, 2022
@author: Lance A. Endres
"""
import numpy                                     as np
from   lendres.geometry.RotationDirection        import RotationDirection
from   lendres.geometry.Shape                    import Shape
from   lendres.mathematics.LinearAlgebra         import AngleIn360Degrees
from   lendres.mathematics.LinearAlgebra         import DiscritizeArc


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


    @property
    def Center(self):
        return self.center


    @property
    def StartPoint(self):
        return self.shapes["start"]


    @property
    def EndPoint(self):
        return self.shapes["end"]


    @property
    def Radius(self):
        return np.linalg.norm(self.shapes["end"].values - self.center.values)


    @property
    def Diameter(self):
        return 2 * self.Radius


    @property
    def Length(self):
        return np.radians(self.Angle) * self.Radius


    @property
    def Angle(self):
        angle = 0
        if self.rotationDirection == RotationDirection.Positive:
            if self.EndAngle < self.StartAngle:
                angle = (360 - self.StartAngle) + self.EndAngle
            else:
                angle = self.EndAngle - self.StartAngle
        else:
            if self.StartAngle < self.EndAngle:
                angle = (360 - self.EndAngle) + self.StartAngle
            else:
                angle = self.StartAngle - self.EndAngle
        return angle


    @property
    def StartAngle(self):
        return AngleIn360Degrees(startPoint=self.center.values, endPoint=self.shapes["start"].values)


    @property
    def EndAngle(self):
        return AngleIn360Degrees(startPoint=self.center.values, endPoint=self.shapes["end"].values)


    def Discritize(self, numberOfPoints=100):
        # Get the starting and ending angles.
        startAngle = AngleIn360Degrees(startPoint=self.center.values, endPoint=self.shapes["start"].values)
        endAngle   = AngleIn360Degrees(startPoint=self.center.values, endPoint=self.shapes["end"].values)

        # The discritize function requires a positive direction.  So we reverse the angles if the the arc is negative.
        if self.rotationDirection == RotationDirection.Negative:
            temp       = endAngle
            endAngle   = startAngle
            startAngle = temp

        points = DiscritizeArc(self.center.values, self.Radius, startAngle, endAngle, numberOfPoints)

        # If the arc goes in the negative direction we have to reverse the points so they come back in the expected order.
        if self.rotationDirection == RotationDirection.Negative:
            points = np.flip(points, axis=0)

        return points