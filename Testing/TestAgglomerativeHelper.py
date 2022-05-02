"""
Created on April 27, 2022
@author: Lance
"""
import DataSetLoading

from   lendres.AgglomerativeHelper      import AgglomerativeHelper

import unittest

class TestAgglomerativeHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCustomerSpendData()


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper                = TestAgglomerativeHelper.dataHelper.Copy(deep=True)
        self.agglomerativeHelper       = AgglomerativeHelper(self.dataHelper, ["Cust_ID", "Name"], copyMethod="exclude")
        self.agglomerativeHelper.ScaleData(method="zscore")


    #def test(self):


    def testBoxPlots(self):
        self.agglomerativeHelper.CreateModel(3)
        self.agglomerativeHelper.FitPredict()
        self.agglomerativeHelper.CreateBoxPlotForClusters()


if __name__ == "__main__":
    unittest.main()