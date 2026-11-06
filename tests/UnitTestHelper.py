"""
Created on April 19, 2022.
@author: Lance A. Endres
"""
import unittest

class UnitTestHelper():
    """
    Class for storing unit test functions.
    """

    @classmethod
    def DiscoverAndRunTests(cls, startDirectory, pattern="test*.py"):
        """
        Discovers all unit tests in the specified directory and runs them.  Tests are discovered based on
        if the file name matches the pattern.

        Parameters
        ----------
        directory : string
            Directory to search for unit tests.
        pattern : string
            Pattern used when searching for unit test files.

        Returns
        -------
        None.
       
        """
        loader   = unittest.TestLoader()
        suite    = loader.discover(startDirectory, pattern=pattern)

        runner   = unittest.TextTestRunner()
        runner.run(suite)