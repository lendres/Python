"""
Created on Febuary 16, 2022
@author: Lance A. Endres
"""
import numpy                                     as np

from   lendres.LinearAlgebra                     import AngleIn360Degrees
import unittest

# More information at:
# https://docs.python.org/3/library/unittest.html

class TestBoundingBinarySearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.points = [1, 3, 5, 8, 11, 14, 18, 22]

    def testAngleIn360Degrees(self):
        result = AngleIn360Degrees([1, 1])
        self.assertAlmostEqual(result, 45.0, 2)

        result = AngleIn360Degrees([-1, 1])
        self.assertAlmostEqual(result, 135.0, 2)

        result = AngleIn360Degrees([-1, -1])
        self.assertAlmostEqual(result, 225.0, 2)

        result = AngleIn360Degrees([1, -1])
        self.assertAlmostEqual(result, 315, 2)

        result = AngleIn360Degrees([-1, -1], returnPositive=False)
        self.assertAlmostEqual(result, -135.0, 2)

        result = AngleIn360Degrees([1, -1], returnPositive=False)
        self.assertAlmostEqual(result, -45, 2)


if __name__ == "__main__":
    unittest.main()