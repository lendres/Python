"""
Created on April 19, 2022.

@author: Lance
"""

import unittest


startDirectory = "C:\\Custom Program Files\\Python\\Testing"

loader   = unittest.TestLoader()
suite    = loader.discover(startDirectory)

runner   = unittest.TextTestRunner()
runner.run(suite)