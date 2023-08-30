"""
Created on November 16, 2022
@author: Lance A. Endres
"""
import DataSetLoading
from   lendres.path.File                                        import File

import os
import unittest


class TestFile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inputFile       = "used_cars_data.csv"
        cls.inputFile   = DataSetLoading.GetFileInDataDirectory(inputFile)


    def testFileSplitAndCombine(self):
        # Split a file into smaller parts.
        outputFiles = File.SplitFileByNumberOfLines(TestFile.inputFile, 2000, True)

        # Recombine the files.  The call to combine the files will also delete the separated files.
        combinedOutputFile = "combined.csv"
        combinedOutputPath = DataSetLoading.GetFileInDataDirectory(combinedOutputFile)
        File.CombineFiles(combinedOutputFile, outputFiles, True)

        # Validate recombination.
        with open(TestFile.inputFile) as originalFile:
            with open(combinedOutputPath) as newFile:
                for originalLine in originalFile:
                    newLine = newFile.readline()
                    self.assertTrue(newLine, "End of new file encountered before it was expected.")
                    self.assertEqual(originalLine, newLine, "The two lines are not equal.")

        # Clean up combined file.
        os.remove(combinedOutputPath)


    def testChangeDirectoryDotDot(self):
        thisDirectory = os.path.dirname(os.path.abspath(__file__))

        solution      = os.path.dirname(thisDirectory)
        result        = File.ChangeDirectoryDotDot(thisDirectory)

        self.assertEqual(solution, result)

        solution      = os.path.dirname(solution)
        result        = File.ChangeDirectoryDotDot(thisDirectory, 2)
        self.assertEqual(solution, result)

        file          = os.path.join(thisDirectory, "file.txt")
        result        = File.ChangeDirectoryDotDot(thisDirectory, 2)
        self.assertEqual(solution, result)



if __name__ == "__main__":
    unittest.main()