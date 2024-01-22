"""
Created on May 27, 2022
@author: Lance A. Endres
"""
from   lendres.path.Path                                             import Path
from   Testing.UnitTestHelper                                        import UnitTestHelper


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Default for running from the hard drive.
startDirectory = Path.GetDirectory(__file__)
UnitTestHelper.DiscoverAndRunTests(startDirectory)