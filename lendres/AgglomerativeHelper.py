"""
Created on April 27, 2022
@author: Lance
"""
import pandas                           as pd
import numpy                            as np
from   matplotlib                       import pyplot                     as plt
import seaborn                          as sns

from   sklearn.cluster                  import AgglomerativeClustering
from   scipy.cluster.hierarchy          import cophenet
from   scipy.cluster.hierarchy          import dendrogram
from   scipy.cluster.hierarchy          import linkage

# Pairwise distribution between data points.
from   scipy.spatial.distance          import pdist

#from lendres.ConsoleHelper            import ConsoleHelper
from   lendres.PlotHelper               import PlotHelper
from   lendres.ClusterHelper            import ClusterHelper


class AgglomerativeHelper(ClusterHelper):

    def __init__(self, dataHelper, columns, copyMethod="include"):
        """
        Constructor.

        Parameters
        ----------
        dataHelper : DataHelper
            DataHelper that has the data in a pandas.DataFrame.

        Returns
        -------
        None.
        """
        super().__init__(dataHelper, columns, copyMethod)


    def CreateModel(self, clusters, distanceMetric="euclidean", linkageMethod="average"):
        self.model = AgglomerativeClustering(n_clusters=clusters, affinity=distanceMetric, linkage=linkageMethod)


    def CreateDendrogramPlot(self, distanceMetric="euclidean", linkageMethod="average", xLabelScale=1.0):
        zLinkages = linkage(self.scaledData, metric=distanceMetric, method=linkageMethod)

        # cophenet index is a measure of the correlation between the distance of points in feature space and distance
        # on dendrogram closer it is to 1, the better is the clustering.
        cophenetCorrelation, cophenetDistances = cophenet(zLinkages , pdist(self.scaledData))

        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot(width=15)

        # The 0.80*PlotHelper.GetScaledStandardSize() is the standard size the PlotHelper uses.
        leafFontSize = 0.80*PlotHelper.GetScaledStandardSize()*xLabelScale
        dendrogram(zLinkages, leaf_rotation=90, color_threshold = 40, leaf_font_size=leafFontSize)

        axis = plt.gca()
        axis.set(title="Agglomerative Hierarchical Clustering Dendogram", xlabel="Sample Index", ylabel="Distance")

        axis.annotate(f"Cophenetic\nCorrelation\n{cophenetCorrelation:0.2f}", (0.90, 0.875), xycoords="axes fraction", fontsize=12*PlotHelper.scale)

        plt.show()


    def GetCophenetCorrelationScores(self):
        # List of distance metrics.
        distanceMetrics = ["chebyshev", "mahalanobis", "cityblock", "euclidean"]

        # List of linkage methods.
        linkageMethods = ["single", "complete", "average", "weighted"]

        columnsLabels = ["Distance Metric", "Linkage Method", "Cophenet Correlation"]
        comparisonFrame = pd.DataFrame(columns=columnsLabels)
        #print("Comparison frame")
        #print(comparisonFrame)

        i = 0
        for distanceMetric in distanceMetrics:
            for linkageMethod in linkageMethods:
                self.AppendCophenetScore(comparisonFrame, i, distanceMetric, linkageMethod)
                i += 1

        # The ward linkage can only use the Euclidean distance.
        self.AppendCophenetScore(comparisonFrame, i, "euclidean", "centroid")
        i += 1
        self.AppendCophenetScore(comparisonFrame, i, "euclidean", "ward")

        # Change the datatype of the correlation column so we can search it for the maximum.
        comparisonFrame["Cophenet Correlation"] = comparisonFrame["Cophenet Correlation"].astype(float)
        indexOfMax                              = comparisonFrame["Cophenet Correlation"].idxmax()

        comparisonFrame.loc["Highest", "Distance Metric"]       = comparisonFrame.loc[indexOfMax, "Distance Metric"]
        comparisonFrame.loc["Highest", "Linkage Method"]        = comparisonFrame.loc[indexOfMax, "Linkage Method"]
        comparisonFrame.loc["Highest", "Cophenet Correlation"]  = comparisonFrame.loc[indexOfMax, "Cophenet Correlation"]

        return comparisonFrame


    def AppendCophenetScore(self, comparisonFrame, row, distanceMetric, linkageMethod):
        zLinkages = linkage(self.scaledData, metric=distanceMetric, method=linkageMethod)
        cophenetCorrelation, cophenetDistances = cophenet(zLinkages, pdist(self.scaledData))

        #thisScoreFrame = pd.DataFrame([[distanceMetric.capitalize(), linkageMethod, cophenetCorrelation]], columns=columnsLabels)
        #print("Score frame")
        #print(thisScoreFrame)
        #pd.concat(comparisonFrame, thisScoreFrame)

        comparisonFrame.loc[row, "Distance Metric"]       = distanceMetric.capitalize()
        comparisonFrame.loc[row, "Linkage Method"]        = linkageMethod
        comparisonFrame.loc[row, "Cophenet Correlation"]  = cophenetCorrelation