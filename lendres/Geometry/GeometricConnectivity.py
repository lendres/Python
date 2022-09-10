"""
Created on Augugst 12, 2022
@author: Lance A. Endres
"""
import numpy                                     as np
from   lendres.geometry.Shape                    import Shape
from   lendres.geometry.Point                    import Point
from   lendres.geometry.Arc                      import Arc
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
        """
        Adds a point to the connectivity.  The connectivity can be searched to see if the point exists already.
        If the search for an existing point is not done, the caller should ensure the point is unique.

        Parameters
        ----------
        point : Point
            Point to add the the connectivity.
        checkForUniqueness : bool, optional
            If true, the existing points are compared for a match to the new point.  If an existing point is
            found, that point is return.  If an existing match is not found, the new point is added and
            returned. The default is False.

        Returns
        -------
        Point
            The point added or found in the connectivity.
        """
        if checkForUniqueness:
            for existingPoint in self.points.values():
                if existingPoint == point:
                    return existingPoint

        self.points[point.id] = point
        return point


    def MakePoint(self, values, checkForUniqueness=False):
        """
        Adds a point to the connectivity created from the input values.  The connectivity can be searched to
        see if a point with those values exists already.  If the search for an existing point is not done,
        the caller should ensure the point is unique.

        Parameters
        ----------
        value : array like
            Location (coordinates) of the point.
        checkForUniqueness : bool, optional
            If true, the existing points are compared for a match to the new point.  If an existing point is
            found, that point is return.  If an existing match is not found, the new point is added and
            returned. The default is False.

        Returns
        -------
        Point
            The point added or found in the connectivity.
        """
        if checkForUniqueness:
            for existingPoint in self.points.values():
                if existingPoint == values:
                    #print("Existing point.", values)
                    return existingPoint

        point = Point(values)
        self.points[point.id] = point
        #print("New point.", values)
        return point


    def GetOrderListOfPoints(self):
        """
        Walks the connectivity to get the points in order.
        """
        outputPoints = []

        startPoint = list(self.points.values())[0]
        outputPoints.append(startPoint.values.tolist())

        shape      = list(startPoint.shapes.values())[0]
        self._InfillShape(shape, startPoint, outputPoints)

        point      = self._WalkToNext(startPoint, shape)

        while point.id != startPoint.id:
            outputPoints.append(point.values.tolist())
            shape = self._WalkToNext(shape, point)
            self._InfillShape(shape, point, outputPoints)
            point = self._WalkToNext(point, shape)

        return outputPoints


    def _WalkToNext(self, startShape, connectiveShape, raiseErrors=True):
        if raiseErrors:
            if len(connectiveShape.shapes.values()) > 2:
                raise Exception("Invalid connectivity found.")

        # Find the next point/shape that we are connected to.
        for shape in connectiveShape.shapes.values():
            if shape.id != startShape.id:
                return shape

        if raiseErrors:
            print("Start shape:", startShape)
            print("Connective shape:", connectiveShape)
            raise Exception("Connectivity not found.")


    def _InfillShape(self, shape, fromPoint, outputPoints):
        if type(shape) == Arc:
            numberOfPoints = int(shape.Length / 0.5)
            points = shape.Discritize(numberOfPoints)
            # Arcs are discritized from start to end.  However, we don't know if we are traversing the arc from
            # start to end or vice versa.  The direction around the shape was chosen at random.  In addition, it is
            # not know how the shapes were created.  Therefore, we need to check to see if we are at the start or end.
            if fromPoint != points[0]:
                points = np.flip(points, axis=0)

            # Don't append the end points.  The main function does that.
            for i in range(1, len(points)-1):
                outputPoints.append(points[i])


    def ConvertToPolygon(self):
        points       = self.GetOrderListOfPoints()
        self.polygon = Polygon(points)