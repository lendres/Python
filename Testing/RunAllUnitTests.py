"""
Created on April 19, 2022.
@author: Lance A. Endres
"""
import unittest

def RunAllTests(startDirectory):
    loader   = unittest.TestLoader()
    suite    = loader.discover(startDirectory)

    runner   = unittest.TextTestRunner()
    runner.run(suite)