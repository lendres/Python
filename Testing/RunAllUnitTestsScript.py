"""
Created on May 27, 2022
@author: Lance A. Endres
"""
from   lendres.path.File                         import File
from   Testing.UnitTestHelper                    import UnitTestHelper
from   lendres.plotting.PlotHelper               import PlotHelper

PlotHelper.formatStyle = "seaborn"

# Default for running from the hard drive.
startDirectory = File.GetDirectory(__file__)
UnitTestHelper.DiscoverAndRunTests(startDirectory)