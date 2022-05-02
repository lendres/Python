"""
Created on April 27, 2022
@author: Lance
"""
import numpy                            as np
from   matplotlib                       import pyplot                     as plt
import seaborn                          as sns

from   sklearn.decomposition            import PCA
from   scipy.cluster.hierarchy          import cophenet
from   scipy.cluster.hierarchy          import dendrogram
from   scipy.cluster.hierarchy          import linkage

# Pairwise distribution between data points.f
from   scipy.spatial.distance          import pdist

#from lendres.ConsoleHelper            import ConsoleHelper
from   lendres.PlotHelper               import PlotHelper
from   lendres.ClusterHelper            import ClusterHelper


class PrincipleComponentAnalysisHelper(ClusterHelper):

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


    def CreateModel(self, clusters):
        self.model = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='average')


    def FitTransform(self):



    def CreateDendrogramPlot(self, metric="euclidean", method="average", xLabelScale=1.0):
        zLinkages = linkage(self.scaledData, metric=metric, method=method)

        # cophenet index is a measure of the correlation between the distance of points in feature space and distance
        # on dendrogram closer it is to 1, the better is the clustering.
        c, coph_dists = cophenet(zLinkages , pdist(self.scaledData))

        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        # The 0.80*PlotHelper.GetScaledStandardSize() is the standard size the PlotHelper uses.
        leafFontSize = 0.80*PlotHelper.GetScaledStandardSize()*xLabelScale
        dendrogram(zLinkages, leaf_rotation=90, color_threshold = 40, leaf_font_size=leafFontSize)

        plt.gca().set(title="Agglomerative Hierarchical Clustering Dendogram", xlabel="Sample Index", ylabel="Distance")

        plt.show()