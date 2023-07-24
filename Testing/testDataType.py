"""
Created on July 23, 2023
@author: Lance A. Endres
"""
import numpy                                     as np

from   lendres.algorithms.DataType               import DataType
import unittest

# More information at:
# https://docs.python.org/3/library/unittest.html

class TestBoundingDataType(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.listOfLists = [[1, 3], [5], [8, 11, 14]]
        cls.mixedList   = [1, [5], [8, 11, 14]]


    def testIsListOfLists(self):
        result = DataType.IsListOfLists(self.listOfLists)
        self.assertTrue(result)

        result = DataType.IsListOfLists(self.mixedList)
        self.assertFalse(result)


    def testContainsAtLeastOneList(self):
        result = DataType.ContainsAtLeastOneList(self.listOfLists)
        self.assertTrue(result)

        result = DataType.ContainsAtLeastOneList(self.mixedList)
        self.assertTrue(result)

        result = DataType.ContainsAtLeastOneList([1, 2, 3])
        self.assertFalse(result)


    def testAreListsOfListsSameSize(self):
        result = DataType.AreListsOfListsSameSize(self.listOfLists, self.listOfLists)
        self.assertTrue(result)

        newList = self.listOfLists.copy()
        newList[0][0] = 0
        result = DataType.AreListsOfListsSameSize(self.listOfLists, self.listOfLists)
        self.assertTrue(result)

        self.assertRaises(Exception, DataType.AreListsOfListsSameSize, self.listOfLists, self.mixedList)


if __name__ == "__main__":
    unittest.main()