"""
Created on April 27, 2022
@author: Lance
"""

import DataSetLoading
from lendres.KMeansHelper           import KMeansHelper

from scipy.stats                    import zscore

import unittest

class TestKMeansHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetTechnicalSupportData()


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper         = TestKMeansHelper.dataHelper.Copy(deep=True)
        self.kMeansHelper       = KMeansHelper(self.dataHelper)


    def testResults(self):
        scaledData = self.dataHelper.data.iloc[:,1:].apply(zscore)
        self.kMeansHelper.CreateElbowPlot(scaledData, range(1, 10))


if __name__ == "__main__":
    unittest.main()