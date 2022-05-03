"""
Created on April 27, 2022
@author: Lance
"""
import numpy                            as np
from   matplotlib                       import pyplot                     as plt
import seaborn                          as sns

from   sklearn.decomposition            import PCA

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

        # Prepend (1, 0) to the front of the cumlative sum to make the plot look correct (start at 0).
        x = np.insert(np.arange(1, len(values)+1), 0, 1, axis=0)
        y = np.insert(np.cumsum(values), 0, 0, axis=0)

        plt.step(x=x, y=y, where="post")
        axis = plt.gca()
        axis.set(title="Cumlative Sum of Variation Explained", xlabel="Eigenvalue", ylabel="Variation Explained")

        # Set the x-ticks so they are only whole numbers at the eigenvalues (nothing in between, no decimal places).
        axis.set_xticks(np.arange(1, len(values)+1))
        plt.show()