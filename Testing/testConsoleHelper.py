"""
Created on December 27, 2021
@author: Lance A. Endres
"""
from   lendres.io.ConsoleHelper                                      import ConsoleHelper
from   lendres.io.IO                                                 import IO

import unittest


skipTests = False

class TestConsoleHelper(unittest.TestCase):
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


    def testIOSingleton(self):
        print("\n\n")
        IO.consoleHelper.Print("Singleton test.")


    def testVerboseLevel(self):
        print("\n\n")
        self.consoleHelper.Print("Standard level.")
        self.consoleHelper.Print("Debug.", ConsoleHelper.VERBOSEDEBUG)
        self.consoleHelper.verboseLevel = ConsoleHelper.VERBOSEDEBUG
        self.consoleHelper.Print("Debug level.", ConsoleHelper.VERBOSEDEBUG)
        self.consoleHelper.verboseLevel = self.verboseLevel


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