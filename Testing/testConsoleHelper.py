"""
Created on December 27, 2021
@author: Lance A. Endres
"""
from   lendres.ConsoleHelper                                    import ConsoleHelper

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


    def testPrintInColor(self):
        self.consoleHelper.PrintInColor("\nThis is a test of foreground color.", (0, 255, 0))
        self.consoleHelper.PrintInColor("This is a test of foreground color and background.", (255, 0, 0), (255, 255, 255))
        self.consoleHelper.Print("Standard print, did the color return to normal?")


if __name__ == "__main__":
    unittest.main()