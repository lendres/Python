"""
Created on December 27, 2021
@author: Lance A. Endres
"""
from   lendres.io.ConsoleHelper                                      import ConsoleHelper
ConsoleHelper().ClearIPythonConsole()
from   lendres.io.IO                                                 import IO
import unittest

class TestTesting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.consoleHelper = ConsoleHelper()
        # cls.consoleHelper.ClearSpyderConsole()
        cls.consoleHelper = ConsoleHelper(verboseLevel=ConsoleHelper.VERBOSEALL)


    def setUp(self):
        pass


    # @unittest.skip
    def testTestOne(self):
        print("\n\n")
        print("Test 1")
        j = 2
        i = 20 + j
        self.assertTrue(i == 22)
        IO.ConsoleHelper.Print("Singleton test.")


    def testTestTwo(self):
        print("\n")
        print("Test 2")
        j = 2
        i = 20 + j
        self.assertTrue(i == 22)
        IO.ConsoleHelper.Print("Singleton test.")


if __name__ == "__main__":
    unittest.main()