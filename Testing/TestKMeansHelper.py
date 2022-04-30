"""
Created on April 27, 2022
@author: Lance
"""
import DataSetLoading
from lendres.KMeansHelper           import KMeansHelper

from scipy.stats                    import zscore
from sklearn.datasets import make_blobs

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


    def testElbowPlot(self):
        scaledData = self.dataHelper.data.iloc[:,1:].apply(zscore)
        self.kMeansHelper.CreateElbowPlot(scaledData, range(1, 10))


    def testSilhouetteAnalysis(self):
        X, y = make_blobs(
            n_samples=500,
            n_features=2,
            centers=4,
            cluster_std=1,
            center_box=(-10.0, 10.0),
            shuffle=True,
            random_state=1,
        )
        self.kMeansHelper.CreateSilhouetteAnalysisPlots(X, range(2, 7))

        scaledData = self.dataHelper.data.iloc[:,2:5].apply(zscore).to_numpy()
        self.kMeansHelper.CreateSilhouetteAnalysisPlots(scaledData, range(2, 7))

if __name__ == "__main__":
    unittest.main()