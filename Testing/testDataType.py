"""
Created on July 23, 2023
@author: Lance A. Endres
"""
import numpy                                                              as np

from   lendres.ConsoleHelper                                              import ConsoleHelper
from   lendres.algorithms.DataType                                        import DataType

import unittest

# More information at:
# https://docs.python.org/3/library/unittest.html

class TestBoundingDataType(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        verboseLevel = ConsoleHelper.VERBOSEREQUESTED
        verboseLevel = ConsoleHelper.VERBOSETESTING
        cls.consoleHelper = ConsoleHelper(verboseLevel=verboseLevel)

        cls.listOfLists  = [[1, 3], [5], [8, 11, 14]]

        # Tuples have to have at least 2 elements.
        cls.listOfTuples = [(1, 3), (5, 6), (8, 11, 14)]

        cls.mixedList    = [1, [5], [8, 11, 14]]


    def testIsListOfLists(self):
        result = DataType.IsListOfLists(self.listOfLists)
        self.assertTrue(result)

        result = DataType.IsListOfLists(self.listOfTuples)
        self.assertTrue(result)

        result = DataType.IsListOfLists(self.mixedList)
        self.assertFalse(result)


    def testContainsAtLeastOneList(self):
        result = DataType.ContainsAtLeastOneList(self.listOfLists)
        self.assertTrue(result)

        result = DataType.ContainsAtLeastOneList(self.listOfTuples)
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


    def testCreateListOfLists(self):
        newListofLists = DataType.CreateListOfLists(self.listOfLists, 1)
        self.consoleHelper.Display(newListofLists, verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        self.assertTrue(DataType.AreListsOfListsSameSize(self.listOfLists, newListofLists))


    def testGetLengthOfNestedObjects(self):
        result = DataType.GetLengthOfNestedObjects(self.listOfLists)
        self.assertEqual(result, 6)

        result = DataType.GetLengthOfNestedObjects(self.mixedList)
        self.assertEqual(result, 5)


if __name__ == "__main__":
    unittest.main()