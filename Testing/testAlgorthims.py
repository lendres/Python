"""
Created on Febuary 16, 2022
@author: Lance A. Endres
"""
import numpy                                     as np

from   lendres.Algorithms                        import BoundingBinarySearch
from   lendres.Algorithms                        import FindIndicesByValues
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



class TestFindIndicesByValues(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        #                0    1    2    3    4    5    6    7    8    9   10   11   12
        cls.numbers = [  1,   3,  11,  11,   5,   8,   5,  11,  14,  18,  22,   3,   3]
        cls.strings = ["a", "b", "c", "b", "b", "d", "d", "e", "a", "a", "b", "b", "b"]


    def testFindNumbers(self):
        result = FindIndicesByValues(TestFindIndicesByValues.numbers, 3)
        self.assertEqual(result[0], 1)
        self.assertEqual(len(result), 3)


    def testFindStrings(self):
        result = FindIndicesByValues(TestFindIndicesByValues.strings, "b")
        self.assertEqual(result[5], 12)
        self.assertEqual(len(result), 6)


    def testFindMaxCount(self):
        result = FindIndicesByValues(TestFindIndicesByValues.strings, "b", maxCount=4)
        self.assertEqual(result[3], 10)
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()