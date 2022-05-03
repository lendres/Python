"""
Created on April 27, 2022
@author: Lance
"""
import pandas                           as pd

import DataSetLoading

from   lendres.ConsoleHelper            import ConsoleHelper
from   lendres.AgglomerativeHelper      import AgglomerativeHelper

from   IPython.display                  import display

import unittest

class TestAgglomerativeHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        verboseLevel = ConsoleHelper.VERBOSEREQUESTED

        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCustomerSpendData(verboseLevel=verboseLevel)
        # Used to display all the columns in the output.
        pd.set_option("display.max_columns", None)

    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper                = TestAgglomerativeHelper.dataHelper.Copy(deep=True)
        self.agglomerativeHelper       = AgglomerativeHelper(self.dataHelper, ["Cust_ID", "Name"], copyMethod="exclude")
        self.agglomerativeHelper.ScaleData(method="zscore")


    def testBoxPlots(self):
        self.agglomerativeHelper.CreateModel(3)
        self.agglomerativeHelper.FitPredict()
        self.agglomerativeHelper.CreateBoxPlotForClusters()


    def testGroupStats(self):
        self.agglomerativeHelper.CreateModel(2)
        self.agglomerativeHelper.FitPredict()
        self.dataHelper.consoleHelper.PrintNewLine(verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        self.dataHelper.consoleHelper.PrintTitle("Group Means", verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        self.dataHelper.consoleHelper.Display(self.agglomerativeHelper.GetGroupMeans(), verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        self.dataHelper.consoleHelper.PrintNewLine(verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        self.dataHelper.consoleHelper.PrintNewLine(verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        self.dataHelper.consoleHelper.PrintTitle("Group Counts", verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        self.dataHelper.consoleHelper.Display(self.agglomerativeHelper.GetGroupCounts(), verboseLevel=ConsoleHelper.VERBOSEREQUESTED)


    def testDendrogramPlot(self):
        self.agglomerativeHelper.CreateDendrogramPlot()
        #self.agglomerativeHelper.CreateDendrogramPlot(distanceMetric="complete", xLabelScale=0.75)
        self.agglomerativeHelper.CreateDendrogramPlot(distanceMetric="euclidean", linkageMethod="ward", xLabelScale=0.75)


    def testCophenetCorrelationScores(self):
        self.dataHelper.consoleHelper.PrintNewLine(verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        self.dataHelper.consoleHelper.PrintNewLine(verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        result = self.agglomerativeHelper.GetCophenetCorrelationScores()
        print(result)


if __name__ == "__main__":
    unittest.main()