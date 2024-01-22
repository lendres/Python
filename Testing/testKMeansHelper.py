"""
Created on April 27, 2022
@author: Lance A. Endres
"""
# Remove in the future.
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import DataSetLoading

import pandas                                                        as pd
from   sklearn.datasets                                              import make_blobs

import unittest

from   lendres.io.ConsoleHelper                                      import ConsoleHelper
from   lendres.DataHelper                                            import DataHelper
from   lendres.KMeansHelper                                          import KMeansHelper

skipTests = 0

class TestKMeansHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        verboseLevel = ConsoleHelper.VERBOSEREQUESTED
        verboseLevel = ConsoleHelper.VERBOSETESTING
        cls.dataHelper, cls.dependentVariable = DataSetLoading.GetTechnicalSupportData(verboseLevel=verboseLevel)


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
        self.dataHelper         = self.dataHelper.Copy()
        self.kMeansHelper       = KMeansHelper(self.dataHelper, ["PROBLEM_TYPE"], copyMethod="exclude")
        self.kMeansHelper.ScaleData(method="standardscaler")

        self.xDataHelper        = self.xDataHelper.Copy()
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
        result = self.kMeansHelper.GetSilhouetteAnalysScores(range(2, 10))
        self.dataHelper.consoleHelper.PrintNewLine(verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        self.dataHelper.consoleHelper.Display(result, verboseLevel=ConsoleHelper.VERBOSEREQUESTED)


    @unittest.skipIf(skipTests, "Skipped box plot test.")
    def testBoxPlots(self):
        result = self.kMeansHelper.GetSilhouetteAnalysScores(range(2, 10))
        self.dataHelper.consoleHelper.Display(result, verboseLevel=ConsoleHelper.VERBOSEREQUESTED)

        self.kMeansHelper.CreateModel(6)
        self.kMeansHelper.FitPredict()
        self.kMeansHelper.CreateBoxPlotsOfClusters("original")
        self.kMeansHelper.CreateBoxPlotsOfClusters("scaled", subPlotColumns=5)


    @unittest.skipIf(skipTests, "Skipped group stats test.")
    def testGroupStats(self):
        self.kMeansHelper.CreateModel(6)
        self.kMeansHelper.FitPredict()
        self.dataHelper.consoleHelper.Display(self.kMeansHelper.GetGroupMeans(), verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        self.dataHelper.consoleHelper.Display(self.kMeansHelper.GetGroupCounts(), verboseLevel=ConsoleHelper.VERBOSEREQUESTED)


if __name__ == "__main__":
    unittest.main()