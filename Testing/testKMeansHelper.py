"""
Created on April 27, 2022
@author: Lance A. Endres
"""
import DataSetLoading
from   lendres.KMeansHelper                      import KMeansHelper
import pandas                                    as pd
import numpy                                     as np


from   sklearn.preprocessing                     import StandardScaler
from   scipy.stats                               import zscore
from   sklearn.datasets                          import make_blobs

from   lendres.DataHelper                        import DataHelper

from   IPython.display                           import display

import unittest

skipTests = 0

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
        cls.xDataHelper = DataHelper(data=pd.DataFrame(X).copy())


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper         = TestKMeansHelper.dataHelper.Copy()
        self.kMeansHelper       = KMeansHelper(self.dataHelper, ["PROBLEM_TYPE"], copyMethod="exclude")
        self.kMeansHelper.ScaleData(method="standardscaler")

        self.xDataHelper        = TestKMeansHelper.xDataHelper.Copy()
        self.xKMeansHelper      = KMeansHelper(self.xDataHelper, [], copyMethod="exclude")
        self.xKMeansHelper.ScaleData(method="zscore")


    def testElbowPlot(self):
        self.kMeansHelper.CreateVisualizerPlot(range(2, 10), metric="distortion")
        self.kMeansHelper.CreateVisualizerPlot((2, 10))


    @unittest.skipIf(skipTests, "Skipped silhouette graphical analysis test.")
    def testSilhouetteGraphicalAnalysis(self):
        data = self.xKMeansHelper.scaledData.to_numpy()
        self.kMeansHelper.CreateTwoColumnSilhouetteVisualizationPlots(data, range(3, 6))
        data = self.kMeansHelper.scaledData.iloc[:, 2:4].to_numpy()
        self.kMeansHelper.CreateTwoColumnSilhouetteVisualizationPlots(data, range(3, 6))


    @unittest.skipIf(skipTests, "Skipped silhouette analysis plots test.")
    def testCreateSilhouetteAnalysisPlots(self):
        self.kMeansHelper.CreateSilhouetteAnalysisPlots(range(3, 6))


    def testSilhouetteScores(self):
        print()
        result = self.kMeansHelper.GetSilhouetteAnalysScores(range(2, 10))
        display(result)


    @unittest.skipIf(skipTests, "Skipped box plot test.")
    def testBoxPlots(self):
        display(self.kMeansHelper.GetSilhouetteAnalysScores(range(2, 10)))
        self.kMeansHelper.CreateModel(6)
        self.kMeansHelper.FitPredict()
        self.kMeansHelper.CreateBoxPlotsOfClusters("original")
        self.kMeansHelper.CreateBoxPlotsOfClusters("scaled", subPlotColumns=5)


    @unittest.skipIf(skipTests, "Skipped group stats test.")
    def testGroupStats(self):
        self.kMeansHelper.CreateModel(6)
        self.kMeansHelper.FitPredict()
        display(self.kMeansHelper.GetGroupMeans())
        display(self.kMeansHelper.GetGroupCounts())


if __name__ == "__main__":
    unittest.main()