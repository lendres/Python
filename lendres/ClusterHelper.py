"""
Created on April 27, 2022
@author: Lance
"""
import pandas                         as pd
import numpy                          as np
from   matplotlib                     import pyplot                     as plt
import matplotlib.cm                  as cm
import matplotlib.ticker              as ticker
import seaborn                        as sns
import math

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


    def GetClusterLabelsAsSeries(self):
        self.dataHelper.data[self.labelColumn]


    def GetClusterCounts(self):
        valueCounts = self.dataHelper.data[self.labelColumn].value_counts()
        valueCounts.sort_index(ascending=True, inplace=True)
        valueCounts.rename("Sample Count", inplace=True)

        countDataFrame            = pd.DataFrame(valueCounts)
        countDataFrame.index.name = "Cluster"
        return countDataFrame


    def GetGroupMeans(self):
        dataFrameOfMeans                 = self.dataHelper.data.groupby([self.labelColumn]).mean()
        dataFrameOfMeans["Sample Count"] = self.dataHelper.data[self.labelColumn].value_counts()
        return dataFrameOfMeans


    def GetGroupedByCluster(self):
        return self.dataHelper.data.groupby([self.labelColumn])


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


    def CreateBoxPlotsOfClusters(self, whichData, subPlotColumns=3):
        """
        Creates a box plot for each cluster.

        Parameters
        ----------
        whichData : string
        subPlotColumns : integer

        Returns
        -------
        None.
        """
        if whichData == "original":
            data = self.dataHelper.data
        elif whichData == "scaled":
            data = self.scaledData
        else:
            raise Exception("The specified data type is invalided.")

        numberOfRows = math.ceil(len(self.columns) / subPlotColumns)
        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot(width=25, height=6*numberOfRows)

        figure, axes = plt.subplots(numberOfRows, subPlotColumns)
        figure.suptitle("Box Plot of Clusters for " + whichData.title() + " Data")

        # Flatten the array (if it is two dimensional) to make it easier to work with and so we
        # don't have to check if it is a one dimensionas (single row of axes) or two dimensional array.
        axes = np.ravel(axes, order="C")

        i = 0
        for column in self.columns:
            if column != self.labelColumn:
                sns.boxplot(ax=axes[i], x=self.dataHelper.data[self.labelColumn], y=data[column])
                i += 1

        # If not all the axes are used, remove the unused.  There will only be empty axis
        # in the last row.
        numberToRemove = subPlotColumns*numberOfRows - len(self.columns)
        j = numberOfRows*subPlotColumns - 1
        for i in range(numberToRemove):
            figure.delaxes(axes[j-i])

        figure.tight_layout()
        plt.show()


    def CreateBarPlotsOfMeanByCluster(self, columns):
        """
        Creates a bar plot of the mean for each cluster.

        Parameters
        ----------
        columns : list of strings
            Columns to plot for each cluster.

        Returns
        -------
        None.
        """
        PlotHelper.FormatPlot()

        if type(columns) != list:
            columns = [columns]

        if not self.labelColumn in columns:
            columns.append(self.labelColumn)

        self.dataHelper.data[columns].groupby(self.labelColumn).mean().plot.bar()
        plt.gca().set_title("Feature Mean by Cluster")
        plt.show()