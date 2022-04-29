"""
Created on April 27, 2022
@author: Lance
"""
import numpy                          as np
from matplotlib                       import pyplot                     as plt

from sklearn.cluster                  import KMeans
from scipy.stats                      import zscore
from scipy.spatial.distance           import cdist

from lendres.ConsoleHelper            import ConsoleHelper
from lendres.PlotHelper               import PlotHelper


class KMeansHelper():

    def __init__(self, dataHelper):
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
        self.dataHelper                = dataHelper


    def CreateElbowPlot(self, data, clusters):
        """
        Constructor.

        Parameters
        ----------
        dataHelper : DataHelper
            DataHelper that has the data in a pandas.DataFrame.
        clusters : list of ints
            The clusters to calculate the distortions for.

        Returns
        -------
        None.
        """
        meanDistortions = []

        for k in clusters:
            model = KMeans(n_clusters=k)
            model.fit(data)
            model.predict(data)

            meanDistortions.append(sum(np.min(cdist(data, model.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        plt.plot(clusters, meanDistortions, "b-", marker="o", markerfacecolor="blue", markersize=12)
        plt.gca().set(title="Selecting the K Value (Number of Clusters) with the Elbow Method", xlabel="K Value", ylabel="Average Distortion")

        figure = plt.gcf()

        plt.show()

        return figure