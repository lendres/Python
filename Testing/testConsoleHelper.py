"""
Created on December 27, 2021
@author: Lance A. Endres
"""
from   lendres.io.ConsoleHelper                                 import ConsoleHelper

import unittest


skipTests = False

class TestDataHelper(unittest.TestCase):
    #verboseLevel = ConsoleHelper.VERBOSENONE
    #verboseLevel = ConsoleHelper.VERBOSETESTING
    #verboseLevel = ConsoleHelper.VERBOSEREQUESTED
    #verboseLevel = ConsoleHelper.VERBOSEIMPORTANT
    verboseLevel = ConsoleHelper.VERBOSEALL

    @classmethod
    def setUpClass(cls):

        cls.consoleHelper = ConsoleHelper(verboseLevel=cls.verboseLevel)


    def setUp(self):
        pass

    def testPrintSpecialMessages(self):
        print("\n\n")
        self.consoleHelper.PrintWarning("This is a warning message.")
        self.consoleHelper.PrintError("This is an error message.")


    def testPrintInColor(self):
        print("\n\n")
        self.consoleHelper.PrintInColor("\nThis is a test of foreground color.", (0, 255, 0))
        self.consoleHelper.PrintInColor("This is a test of foreground and background color.", (255, 0, 0), (255, 255, 255))
        self.consoleHelper.Print("Standard print, did the color return to normal?")


if __name__ == "__main__":
    unittest.main()