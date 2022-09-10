"""
Created on April 27, 2022
@author: Lance A. Endres
"""
import pandas                                    as pd
from   matplotlib                                import pyplot                     as plt

from   sklearn.cluster                           import AgglomerativeClustering
from   scipy.cluster.hierarchy                   import cophenet
from   scipy.cluster.hierarchy                   import dendrogram
from   scipy.cluster.hierarchy                   import linkage

# Pairwise distribution between data points.
from   scipy.spatial.distance                    import pdist

from   lendres.plotting.PlotHelper              import PlotHelper
from   lendres.ClusterHelper                     import ClusterHelper


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
        self.distanceMetric = "none"
        self.linkageMethod  = "none"

        super().__init__(dataHelper, columns, copyMethod)


    def CreateModel(self, clusters, distanceMetric="euclidean", linkageMethod="average"):
        self.distanceMetric = distanceMetric
        self.linkageMethod  = linkageMethod
        self.model = AgglomerativeClustering(n_clusters=clusters, affinity=distanceMetric, linkage=linkageMethod)


    def CreateMyDendrogramPlot(self, xLabelScale=1.0, cutDistance=None, drawCutLine=False):
        self.CreateDendrogramPlot(
            distanceMetric=self.distanceMetric,
            linkageMethod=self.linkageMethod,
            xLabelScale=xLabelScale,
            cutDistance=cutDistance,
            drawCutLine=drawCutLine
        )


    def CreateDendrogramPlot(self, distanceMetric="euclidean", linkageMethod="average", xLabelScale=1.0, cutDistance=None, drawCutLine=False):
        linkageDistances = linkage(self.scaledData, metric=distanceMetric, method=linkageMethod)

        # cophenet index is a measure of the correlation between the distance of points in feature space and distance
        # on dendrogram closer it is to 1, the better is the clustering.
        cophenetCorrelation, cophenetDistances = cophenet(linkageDistances , pdist(self.scaledData))

        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot(width=15)

        # The 0.80*PlotHelper.GetScaledStandardSize() is the standard size the PlotHelper uses.  We scale that
        # by the argument provided.
        leafFontSize = 0.80*PlotHelper.GetScaledStandardSize()*xLabelScale
        dendrogram(linkageDistances, leaf_rotation=90, color_threshold=cutDistance, leaf_font_size=leafFontSize)

        # Final formating.
        # Main title.
        axis  = plt.gca()
        title = "Agglomerative Hierarchical Clustering Dendogram\n"
        axis.set(title=title, xlabel="Sample Index", ylabel="Distance")

        # Subtitle.
        title = "Distance Metric = " + distanceMetric.capitalize() + ", Linkage Method = " + linkageMethod.capitalize()
        plt.suptitle(title, fontsize=0.9*PlotHelper.scale*PlotHelper.size)

        # Cophenetic score annotation.
        axis.annotate(f"Cophenetic\nCorrelation\n{cophenetCorrelation:0.3f}", (0.90, 0.875), xycoords="axes fraction", fontsize=13*PlotHelper.scale)

        # Cut line.
        if drawCutLine:
            axis.axhline(y=cutDistance, c="red", lw=1.5*PlotHelper.scale, linestyle="dashdot")

        plt.show()


    def GetCophenetCorrelationScores(self):
        # List of distance metrics.
        distanceMetrics = ["chebyshev", "mahalanobis", "cityblock", "euclidean"]

        # List of linkage methods.
        linkageMethods = ["single", "complete", "average", "weighted"]

        # Create a DataFrame to store the results.
        columnsLabels = ["Distance Metric", "Linkage Method", "Cophenet Correlation"]
        comparisonFrame = pd.DataFrame(columns=columnsLabels)
        #print("Comparison frame")
        #print(comparisonFrame)

        i = 0
        for distanceMetric in distanceMetrics:
            for linkageMethod in linkageMethods:
                self.AppendCophenetScore(comparisonFrame, i, distanceMetric, linkageMethod)
                i += 1

        # The centroid and ward linkage can only use the Euclidean distance.
        self.AppendCophenetScore(comparisonFrame, i, "euclidean", "centroid")
        i += 1
        self.AppendCophenetScore(comparisonFrame, i, "euclidean", "ward")

        # Change the datatype of the correlation column so we can search it for the maximum.
        comparisonFrame["Cophenet Correlation"] = comparisonFrame["Cophenet Correlation"].astype(float)
        indexOfMax                              = comparisonFrame["Cophenet Correlation"].idxmax()

        # Add the maximum value as the last entry.
        comparisonFrame.loc["Highest", "Distance Metric"]       = comparisonFrame.loc[indexOfMax, "Distance Metric"]
        comparisonFrame.loc["Highest", "Linkage Method"]        = comparisonFrame.loc[indexOfMax, "Linkage Method"]
        comparisonFrame.loc["Highest", "Cophenet Correlation"]  = comparisonFrame.loc[indexOfMax, "Cophenet Correlation"]

        return comparisonFrame


    def AppendCophenetScore(self, comparisonFrame, row, distanceMetric, linkageMethod):
        linkageDistances = linkage(self.scaledData, metric=distanceMetric, method=linkageMethod)
        cophenetCorrelation, cophenetDistances = cophenet(linkageDistances, pdist(self.scaledData))

        #thisScoreFrame = pd.DataFrame([[distanceMetric.capitalize(), linkageMethod, cophenetCorrelation]], columns=columnsLabels)
        #print("Score frame")
        #print(thisScoreFrame)
        #pd.concat(comparisonFrame, thisScoreFrame)

        comparisonFrame.loc[row, "Distance Metric"]       = distanceMetric
        comparisonFrame.loc[row, "Linkage Method"]        = linkageMethod
        comparisonFrame.loc[row, "Cophenet Correlation"]  = cophenetCorrelation