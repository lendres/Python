"""
Created on April 27, 2022
@author: Lance
"""
import numpy                            as np
from   matplotlib                       import pyplot                     as plt
import matplotlib.cm                    as cm
import matplotlib.ticker                as ticker
import seaborn                          as sns

from   sklearn.cluster                  import AgglomerativeClustering
from   scipy.stats                      import zscore
from   scipy.spatial.distance           import cdist

from   sklearn.metrics                  import silhouette_samples
from   sklearn.metrics                  import silhouette_score

# To visualize the elbow curve and silhouette scores.
from   yellowbrick.cluster              import KElbowVisualizer
from   yellowbrick.cluster              import SilhouetteVisualizer

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


    def CreateModel(self, clusters):
        self.model = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='average')