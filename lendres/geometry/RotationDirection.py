"""
Created on August 13, 2022
@author: Lance A. Endres
"""
from   enum                                      import IntEnum
from   enum                                      import auto


class RotationDirection(IntEnum):
    # Same as counter-clockwise in most cases.
    Positive         = 0

    # Same as clockwise in most cases.
    Negative         = auto()

    # Used to denote the size of the enum.
    # Can be used to create arrays of the proper size or in loops.
    End              = auto()