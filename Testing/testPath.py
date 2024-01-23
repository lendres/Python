"""
Created on November 16, 2022
@author: Lance A. Endres
"""
import DataSetLoading
from   lendres.path.Path                                             import Path

import os
import unittest


class TestPath(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inputFile       = "used_cars_data.csv"
        cls.inputFile   = DataSetLoading.GetFileInDataDirectory(inputFile)


    def testChangeDirectoryDotDot(self):
        thisDirectory = os.path.dirname(os.path.abspath(__file__))

        solution      = os.path.dirname(thisDirectory)
        result        = Path.ChangeDirectoryDotDot(thisDirectory)

        self.assertEqual(solution, result)

        solution      = os.path.dirname(solution)
        result        = Path.ChangeDirectoryDotDot(thisDirectory, 2)
        self.assertEqual(solution, result)

        result        = Path.ChangeDirectoryDotDot(thisDirectory, 2)
        self.assertEqual(solution, result)


    def testContainsDirectory(self):
        fileName = "c:/temp/test.txt"
        result   = Path.ContainsDirectory(fileName)
        self.assertTrue(result)

        fileName = "c:\\temp\\test.txt"
        result   = Path.ContainsDirectory(fileName)
        self.assertTrue(result)

        fileName = "test.txt"
        result   = Path.ContainsDirectory(fileName)
        self.assertFalse(result)


    def testGetDirectory(self):
        fileName = "c:/temp/test.txt"
        result   = Path.GetDirectory(fileName)
        self.assertEqual("c:\\temp", result)

        fileName = "c:\\temp\\test.txt"
        result   = Path.GetDirectory(fileName)
        self.assertEqual("c:\\temp", result)

        # If no directory is provied, the current path will be used.
        thisDirectory = os.path.dirname(os.path.abspath(__file__))
        fileName      = "test.txt"
        result        = Path.GetDirectory(fileName)
        self.assertEqual(thisDirectory, result.lower())


if __name__ == "__main__":
    unittest.main()