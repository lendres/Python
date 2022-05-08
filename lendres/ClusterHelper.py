"""
Created on April 27, 2022
@author: Lance
"""
import numpy                          as np
from   matplotlib                     import pyplot                     as plt
import matplotlib.cm                  as cm
import matplotlib.ticker              as ticker
import seaborn                        as sns

from   scipy.spatial.distance         import cdist

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


    def GetLabelSeries(self):
        self.dataHelper.data[self.labelColumn]


    def GetGroupMeans(self):
        return self.dataHelper.data.groupby([self.labelColumn]).mean()


    def GetGroupCounts(self):
        return self.dataHelper.data.groupby([self.labelColumn]).sum()


    def DisplayValueCountsByCluster(self, column):
        """
        Displays the value counts for the specified column as they are grouped the clusterer.

        Parameters
        ----------
        column : string
            Column to display the value counts for.

        Returns
        -------
        None.
        """
        numberOfClusters = self.model.n_clusters

        for i in range(numberOfClusters):

            result = self.dataHelper.data[self.dataHelper.data[self.labelColumn] == i][column].value_counts()

            self.dataHelper.consoleHelper.PrintTitle("Cluster " + str(i))
            self.dataHelper.consoleHelper.Print(result)


    def CreateBoxPlotForClusters(self):
        """
        Creates a box plot for each cluster.

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
            if column != self.labelColumn:
                sns.boxplot(x=self.dataHelper.data[self.labelColumn], y=self.dataHelper.data[column])
                plt.show()