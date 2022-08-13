"""
Created on Augugst 12, 2022
@author: Lance A. Endres
"""
from   lendres.Geometry.Shape                    import Shape
from   lendres.Geometry.Point                    import Point
from   lendres.Geometry.Arc                      import Arc
from   shapely.geometry                          import Polygon


class GeometricConnectivity(Shape):


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
        super().__init__()

        self.points  = {}
        self.polygon = None


    def AddPoint(self, point, checkForUniqueness=False):
        if checkForUniqueness:
            for existingPoint in self.points.values():
                if existingPoint.EqualsPoint(point):
                    return existingPoint

        self.points[point.id] = point
        return point


    def MakePoint(self, values, checkForUniqueness=False):
        if checkForUniqueness:
            for existingPoint in self.points.values():
                if existingPoint.EqualsValues(values):
                    print("Existing point.", values)
                    return existingPoint

        point = Point(values)
        self.points[point.id] = point
        print("New point.", values)
        return point


    def GetOrderListOfPoints(self):
        """
        Walks the connectivity to get the points in order.
        """
        outputPoints = []

        startPoint = list(self.points.values())[0]
        outputPoints.append(startPoint.values.tolist())

        shape      = list(startPoint.shapes.values())[0]
        self._InfillShape(shape, outputPoints)

        point      = self._WalkToNext(startPoint, shape)

        while point.id != startPoint.id:
            outputPoints.append(point.values.tolist())
            shape = self._WalkToNext(shape, point)
            self._InfillShape(shape, outputPoints)
            point = self._WalkToNext(point, shape)

        return outputPoints


    def _WalkToNext(self, startShape, connectiveShape):
        for shape in connectiveShape.shapes.values():
            if shape.id != startShape.id:
                return shape


    def _InfillShape(self, shape, outputPoints):
        if type(shape) == Arc:
            points = shape.Discritize(50)

            # Don't append the end points.  The main function does that.
            for i in range(1, len(points)-1):
                outputPoints.append(points[i])


    def ConvertToPolygon(self):
        points       = self.GetOrderListOfPoints()
        self.polygon = Polygon(points)