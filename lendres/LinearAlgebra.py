"""
Created on Thu August 11, 2022
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