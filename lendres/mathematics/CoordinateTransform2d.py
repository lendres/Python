"""
Created on November 3, 2024
@author: Lance A. Endres
"""
import numpy as np


class CoordinateTransform2d():
    """
    Two dimensional coordinate transformation (translation and rotation).

    The rotation is applied before the translation.
    """


    def __init__(self, translation:list=[0, 0], angle:float=0, degrees:bool=False):
        """
        Intialize with a transformation and a rotation.

        Parameters
        ----------
        translation : list, optional
            The translation amount. The default is [0, 0].
        angle : float, optional
            The angle of rotation. The default is 0.
        degrees : bool, optional
            If True, "angle" is assumed to be in degrees, otherwise, angle is assumed to be in radians. The default is False.

        Returns
        -------
        None.
        """
        self.translation = translation
        self.CreateRotation(angle, degrees)


    @property
    def Translation(self):
        return self.translation


    @Translation.setter
    def Translation(self, translation:list):
        self.translation = translation


    def CreateRotation(self, angle:float, degrees:bool=False) -> list:
        """
        Creates a 2D rotation matrix.

        Parameters
        ----------
        angle : float
            The angle of rotation.
        degrees : bool, optional
            If True, "angle" is assumed to be in degrees, otherwise, angle is assumed to be in radians. The default is False.

        Returns
        -------
        list
            The 2x2 "matrix" that represents a rotation in 2d.
        """
        if degrees:
            angle = np.radians(angle)

        self.rotationMatrix = [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ]

        return self.rotationMatrix


    def Apply(self, points:list|tuple|np.ndarray) -> np.ndarray:
        """
        Apply the coordinate transformation to a point or set of points.

        Parameters
        ----------
        points : list|tuple|np.ndarray
            Points to apply the transformation to.

        Returns
        -------
        points : numpy.ndarray
            The point or points that have been translated and rotated.
        """
        if type(points) is list or type(points) is tuple:
            points = np.array(points)

        match len(points.shape):
            case 1:
                # Points has only one dimension, so a single point should have been passed in.
                if len(points) != 2:
                    raise Exception("Invalid shape of input points.")
                length = 1
            case 2:
                if points.shape[1] != 2:
                    raise Exception("Invalid shape of input points.")
                length = points.shape[0]
            case _:
                raise Exception("Invalid shape of input points.")

        if length == 1:
            points = self.__Transform(points)
        else:
            points = [self.__Transform(point) for point in points]

        return points


    def __Transform(self, point:list|tuple|np.ndarray) -> np.ndarray:
        """
        Applies the transformation to one point.

        Parameters
        ----------
        point : list|tuple|np.ndarray
            The point to transform.

        Returns
        -------
        numpy.ndarray
            The point after translation and rotation.
        """
        return np.matmul(self.rotationMatrix, point) + self.translation