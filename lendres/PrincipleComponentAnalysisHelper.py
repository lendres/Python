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
from   lendres.SubsetHelper             import SubsetHelper


class PrincipleComponentAnalysisHelper(SubsetHelper):

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


    def CreateModel(self, components="all"):
        # If "all" is specified, the number of clusters is the number of columns.
        if components == "all":
            components = self.scaledData.shape[1]

        self.model = PCA(n_components=components)


    def Fit(self):
        self.model.fit(self.scaledData)


    def FitTransform(self):
        return self.model.fit_transform(self.scaledData)


    def CreateVarianceExplainedPlot(self):

        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        values  = self.model.explained_variance_ratio_
        xlabels = np.arange(1, len(values)+1)
        sns.barplot(x=xlabels, y=values, palette="winter")

        plt.gca().set(title="Variation Explained by Eigenvalue", xlabel="Eigenvalue", ylabel="Variation Explained")
        plt.show()

        plt.step(xlabels, np.cumsum(values), where="pre")
        plt.gca().set(title="Cumlative Sum of Variation Explained", xlabel="Eigenvalue", ylabel="Variation Explained")
        plt.show()