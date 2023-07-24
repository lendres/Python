"""
Created on July 20, 2023
@author: Lance A. Endres
"""
import matplotlib.pyplot                                        as plt
import os

from   lendres.plotting.AxesHelper                              import AxesHelper
from   lendres.plotting.PlotMaker                               import PlotMaker
from   lendres.data.DataComparison                              import DataComparison

import unittest

class TestDataComparison(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        thisDirectory = os.path.dirname(os.path.abspath(__file__))

        cls.dispColumn  = "bit_disp_cumulate (deg)"
        cls.velColumn   = "w_bit"

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
        dataSet["Displacement (revs)"] = dataSet["bit_disp_cumulate (deg)"] / 360.0


    def testGetData(self):
        self.assertEqual(self.dataComparison.dataSets[0].shape[0], 60002)
        self.assertEqual(self.dataComparison.dataSets[1].shape[0], 6001)


    def testEndTime(self):
        self.assertEqual(self.dataComparison.GetEndTime(), 60.0)


    def testPlotComparison(self):
        self.dataComparison.CreateComparisonPlot("Displacement (revs)")


    def testCreateDualAxisComparisonPlot(self):
        columns = [["Displacement (revs)"], ["w_bit"]]
        labels = ["Displacement (revs)", "Velocity (rpm)"]
        self.dataComparison.CreateDualAxisComparisonPlot(columns, labels)


    def testGetVelocity(self):
        self.assertAlmostEqual(self.dataComparison.GetValueAtTime(0, self.velColumn, 11), 38.4657299, places=3)
        self.assertAlmostEqual(self.dataComparison.GetVelocityAtTime(1, self.velColumn, 11), 75.1645423, places=3)


if __name__ == "__main__":
    unittest.main()