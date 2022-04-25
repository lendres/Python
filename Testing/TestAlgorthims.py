"""
Created on Febuary 16, 2022
@author: Lance
"""
import numpy as np

from lendres.Algorithms import BoundingBinarySearch
import unittest

# More information at:
# https://docs.python.org/3/library/unittest.html


class TestBoundingBinarySearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.points = [1, 3, 5, 8, 11, 14, 18, 22]

    def testIndicesFirstHalf(self):
        result = BoundingBinarySearch(2, TestBoundingBinarySearch.points, returnedUnits="indices")
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 1)


    def testIndicesSecondHalf(self):
        result = BoundingBinarySearch(16, TestBoundingBinarySearch.points, returnedUnits="indices")
        self.assertEqual(result[0], 5)
        self.assertEqual(result[1], 6)


    def testIndicesOnAPoint(self):
        result = BoundingBinarySearch(5, TestBoundingBinarySearch.points, returnedUnits="indices")
        self.assertEqual(result[0], 2)
        self.assertEqual(result[1], 2)


    def testValues(self):
        result = BoundingBinarySearch(2, TestBoundingBinarySearch.points, returnedUnits="values")
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 3)


    def testValueTooLow(self):
        result = BoundingBinarySearch(-2, TestBoundingBinarySearch.points, returnedUnits="values")
        self.assertTrue(result[0] is np.nan)
        self.assertTrue(result[1] is np.nan)


    def testValueTooHigh(self):
        result = BoundingBinarySearch(10000, TestBoundingBinarySearch.points, returnedUnits="values")
        self.assertTrue(result[0] is np.nan)
        self.assertTrue(result[1] is np.nan)


if __name__ == "__main__":
    unittest.main()