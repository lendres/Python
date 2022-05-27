"""
Created on April 19, 2022.
@author: Lance
"""

import unittest


def RunAllTests(startDirectory):
    loader   = unittest.TestLoader()
    suite    = loader.discover(startDirectory)

    runner   = unittest.TextTestRunner()
    runner.run(suite)


# Default for running from the hard drive.
startDirectory = "C:\\Custom Program Files\\Python\\Testing"
RunAllTests(startDirectory)