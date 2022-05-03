"""
Created on April 27, 2022
@author: Lance
"""
import numpy                          as np
from   matplotlib                     import pyplot                     as plt
import matplotlib.cm                  as cm
import matplotlib.ticker              as ticker
import seaborn                        as sns

from   sklearn.preprocessing          import StandardScaler
from   scipy.stats                    import zscore
from   scipy.spatial.distance         import cdist

from   sklearn.metrics                import silhouette_samples
from   sklearn.metrics                import silhouette_score

# To visualize the elbow curve and silhouette scores.
from   yellowbrick.cluster            import SilhouetteVisualizer

#from   lendres.ConsoleHelper          import ConsoleHelper
from   lendres.PlotHelper             import PlotHelper
from   lendres.SubsetHelper           import SubsetHelper


class ClusterHelper(SubsetHelper):

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


    def FitPredict(self):
        self.model.fit_predict(self.scaledData)
        self.LabelData()


    def LabelData(self):
        self.dataHelper.data[self.labelColumn] = self.model.labels_


    def GetGroupMeans(self):
        return self.dataHelper.data.groupby([self.labelColumn]).mean()


    def GetGroupCounts(self):
        return self.dataHelper.data.groupby([self.labelColumn]).sum()


    def CreateBoxPlotForClusters(self):
        """
        Constructor.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        for column in self.columns:
            sns.boxplot(x=self.dataHelper.data[self.labelColumn], y=self.dataHelper.data[column])
            plt.show()