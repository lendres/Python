"""
Created on May 27, 2022
@author: Lance A. Endres
"""
from   lendres.path.File                         import File
from   Testing.UnitTestHelper                    import UnitTestHelper

# Default for running from the hard drive.
startDirectory = File.GetDirectory(__file__)
UnitTestHelper.DiscoverAndRunTests(startDirectory)