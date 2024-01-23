"""
Created on Febuary 16, 2022
@author: Lance A. Endres
"""
import matplotlib.pyplot                                        as plt

from   lendres.mathematics.Angles                               import AngleIn360Degrees
from   lendres.mathematics.Angles                               import DiscritizeArc

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


    def testDiscritizeArc(self):
        points = DiscritizeArc(center=[0, 0], radius=1, startAngle=0, endAngle=90, numberOfPoints=100)
        plt.plot(points[:, 0], points[:, 1])
        plt.show()

        points = DiscritizeArc(center=[0, 0], radius=1, startAngle=45, endAngle=135, numberOfPoints=100)
        plt.plot(points[:, 0], points[:, 1])
        plt.show()

        points = DiscritizeArc(center=[1, 1], radius=1, startAngle=180, endAngle=270, numberOfPoints=100)
        plt.plot(points[:, 0], points[:, 1])
        plt.show()

        points = DiscritizeArc(center=[1, 1], radius=1, startAngle=235, endAngle=35, numberOfPoints=100)
        plt.plot(points[:, 0], points[:, 1])
        plt.show()

        points = DiscritizeArc(center=[1, 1], radius=1, startAngle=0, endAngle=180, numberOfPoints=100)
        plt.plot(points[:, 0], points[:, 1])
        plt.show()

        points = DiscritizeArc(center=[1, 1], radius=1, startAngle=180, endAngle=0, numberOfPoints=100)
        plt.plot(points[:, 0], points[:, 1])
        plt.show()




if __name__ == "__main__":
    unittest.main()