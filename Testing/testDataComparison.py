"""
Created on July 20, 2023
@author: Lance A. Endres
"""
import pandas                                                        as pd
import os
import math

from   lendres.io.ConsoleHelper                                      import ConsoleHelper
from   lendres.data.DataComparison                                   import DataComparison

import unittest

pd.set_option('display.max_columns', None)

class TestDataComparison(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        verboseLevel = ConsoleHelper.VERBOSEREQUESTED
        verboseLevel = ConsoleHelper.VERBOSETESTING
        cls.consoleHelper = ConsoleHelper(verboseLevel=verboseLevel)

        thisDirectory = os.path.dirname(os.path.abspath(__file__))

        cls.dispColumn    = "bit_disp_cumulate (rad)"
        cls.velColumn     = "w_bit"

        cls.dataComparison = DataComparison(
            directory           = os.path.join(thisDirectory, "Data"),
            independentColumn   = "Time"

        )

        cls.dataComparison.LoadFile("dynamicsmodel1.csv", "Model 1")
        cls.dataComparison.LoadFile("dynamicsmodel2.csv", "Model 2")


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataComparison.Apply(self.AddRevolutions)


    def AddRevolutions(self, dataSet):
        self.consoleHelper.Display(dataSet.head(), verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        dataSet["Displacement (revs)"] = dataSet["bit_disp_cumulate (rad)"] / 2 / math.pi


    def testGetData(self):
        self.assertEqual(self.dataComparison.dataSets[0].shape[0], 60002)
        self.assertEqual(self.dataComparison.dataSets[1].shape[0], 6001)


    def testEndTime(self):
        self.assertEqual(self.dataComparison.GetEndTime(), 60.0)


    def testPlotComparison(self):
        # Test basic plot.
        self.dataComparison.CreateComparisonPlot("Displacement (revs)")

        # Test supplying axes labels.
        self.dataComparison.CreateComparisonPlot("Displacement (revs)", xLabel="Time (s)", yLabel="Displacement (revolutions)")

        # Test supplying multiple columns.
        self.dataComparison.CreateComparisonPlot(["w_td", "w_bit"], xLabel="Time (s)", yLabel="Displacement (revolutions)")


    def testCreateDualAxisComparisonPlot(self):
        columns = [["Displacement (revs)"], ["w_bit"]]
        labels = ["Displacement (revs)", "Velocity (rpm)"]
        self.dataComparison.CreateMultiAxisComparisonPlot(columns, labels)


    def testGetVelocity(self):
        self.assertAlmostEqual(self.dataComparison.GetValue(0, self.velColumn, 11), 38.4657299, places=3)
        self.assertAlmostEqual(self.dataComparison.GetValue(1, self.velColumn, 11), 75.1645423, places=3)


if __name__ == "__main__":
    unittest.main()