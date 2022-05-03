"""
Created on April 27, 2022
@author: Lance
"""
import DataSetLoading
from   lendres.KMeansHelper             import KMeansHelper
import pandas                           as pd
import numpy                            as np


from   sklearn.preprocessing            import StandardScaler
from   scipy.stats                      import zscore
from   sklearn.datasets                 import make_blobs

from   lendres.DataHelper               import DataHelper

from   IPython.display                  import display

import unittest

class TestKMeansHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetTechnicalSupportData()


        X, y = make_blobs(n_samples=500,
                          n_features=2,
                          centers=4,
                          cluster_std=1,
                          center_box=(-10.0, 10.0),
                          shuffle=True,
                          random_state=1
                         )
        cls.xDataHelper = DataHelper(data=pd.DataFrame(X), copy=True, deep=True)


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper         = TestKMeansHelper.dataHelper.Copy(deep=True)
        self.kMeansHelper       = KMeansHelper(self.dataHelper, ["PROBLEM_TYPE"], copyMethod="exclude")
        self.kMeansHelper.ScaleData(method="standardscaler")

        self.xDataHelper        = TestKMeansHelper.xDataHelper.Copy(deep=True)
        self.xKMeansHelper      = KMeansHelper(self.xDataHelper, [], copyMethod="exclude")
        self.xKMeansHelper.ScaleData(method="zscore")


    def testElbowPlot(self):
        self.kMeansHelper.CreateElbowPlot(range(2, 10))
        self.kMeansHelper.CreateElbowPlot2((2, 10))


    def testSilhouetteGraphicalAnalysis(self):
        data = self.xKMeansHelper.scaledData
        self.kMeansHelper.CreateTwoColumnSilhouetteVisualizationPlots(data, range(3, 6))
        data = self.kMeansHelper.scaledData[:, 2:4]
        self.kMeansHelper.CreateTwoColumnSilhouetteVisualizationPlots(data, range(3, 6))


    def testCreateSilhouetteAnalysisPlots(self):
        self.kMeansHelper.CreateSilhouetteAnalysisPlots(range(3, 6))


    def testSilhouetteScores(self):
        print()
        self.kMeansHelper.DisplaySilhouetteAnalysScores(range(2, 10))


    def testBoxPlots(self):
        self.kMeansHelper.DisplaySilhouetteAnalysScores(range(2, 10))
        self.kMeansHelper.CreateModel(6)
        self.kMeansHelper.FitPredict()
        self.kMeansHelper.CreateBoxPlotForClusters()


    def testGroupStats(self):
        self.kMeansHelper.CreateModel(6)
        self.kMeansHelper.FitPredict()
        display(self.kMeansHelper.GetGroupMeans())
        display(self.kMeansHelper.GetGroupCounts())


if __name__ == "__main__":
    unittest.main()