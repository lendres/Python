"""
Created on August 11, 2022
@author: Lance A. Endres
"""
import numpy                                     as np


# create function to compute angle
def AngleIn360Degrees(endPoint, startPoint=[0, 0], returnPositive=True):
    """
    Calculates the angle (0 to 360) a point or line is at.

    Parameters
    ----------
    startPoint : array like
        Line starting point.  If none is provide, it is assumed to be the origin.
    endPoint : array like
        Line ending point.

    Returns
    -------
    angle : double
        Angle between 0 and 360 degrees.
    """
    # Translate line/point to the origin.
    p1 = np.array(endPoint) - np.array(startPoint)

    angle = np.degrees(np.arctan2(p1[1], p1[0]))

    if returnPositive:
        angle = angle % 360.0

    return angle


def DiscritizeArc(center, radius:float, startAngle:float, endAngle:float, numberOfPoints:int):
    """
    Creates a discritized arc.  Useful for plotting of discritized calculations.

    Works by creating the arc with the at [0, 0] and the start angle 0 degrees (positive x-axis).  The
    arc is then rotated and translated to the requested position.

    Parameters
    ----------
    center : TYPE
        DESCRIPTION.
    radius : float
        Arc radius.
    startAngle : float
        Arc start angle.
    endAngle : float
        Arc end angle.
    numberOfPoints : int
        The number of points to generate along the arc.

    Returns
    -------
    points : ndarray
        Points along the arc.
    """
    # Calculate the swept angle.
    arcAngle = 360 - startAngle + endAngle if endAngle < startAngle else endAngle - startAngle
    arcAngle = np.radians(arcAngle)

    # Get a set of angles (discritizes the total angle).
    angles = np.linspace(0, arcAngle, numberOfPoints)

    # Initialize output.
    points = np.zeros((numberOfPoints, 2))

    points[:, 0] = radius * np.cos(angles)
    points[:, 1] = radius * np.sin(angles)

    # Create a rotation matrix.
    startAsRadians = np.radians(startAngle)
    rotationMatrix = [
        [np.cos(startAsRadians),  -np.sin(startAsRadians)],
        [np.sin(startAsRadians),   np.cos(startAsRadians)]
    ]

    # Perform the translation and rotation to the points.
    for i in range(numberOfPoints):
        # Rotation.
        result = np.matmul(rotationMatrix, points[i])

        # Translation.
        result = result + center

        points[i, :] = result

    return points