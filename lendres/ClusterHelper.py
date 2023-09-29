"""
Created on April 27, 2022
@author: Lance
"""
import pandas                                                   as pd
import numpy                                                    as np
from   matplotlib                                               import pyplot                     as plt
import seaborn                                                  as sns
import math

from   lendres.plotting.PlotHelper                              import PlotHelper
from   lendres.SubsetHelper                                     import SubsetHelper


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
        """
        Fits a hyperparameter search and runs predict.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.model.fit_predict(self.scaledData)
        self.LabelData()


    def LabelData(self):
        """
        Adds the labels from the fitted model to the DataFrame containing the data.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.dataHelper.data[self.labelColumn] = self.model.labels_


    def GetClusterLabelsAsSeries(self):
        """
        Gets cluster labels for each sample in the data.

        Parameters
        ----------
        None.

        Returns
        -------
        : Series
            Series containing the labels for each sample.
        """
        self.dataHelper.data[self.labelColumn]


    def GetClusterCounts(self):
        """
        Gets the counts of elements in each cluster.

        Parameters
        ----------
        None.

        Returns
        -------
        countDataFrame : DataFrame
            DataFrame containing the count of each group/cluster.
        """
        valueCounts = self.dataHelper.data[self.labelColumn].value_counts()
        valueCounts.sort_index(ascending=True, inplace=True)
        valueCounts.rename("Sample Count", inplace=True)

        countDataFrame            = pd.DataFrame(valueCounts)
        countDataFrame.index.name = "Cluster"
        return countDataFrame


    def GetGroupMeans(self):
        """
        Gets the mean of each cluster.

        Parameters
        ----------
        None.

        Returns
        -------
        dataFrameOfMeans : DataFrame
            DataFrame containing the mean of each group/cluster.
        """
        dataFrameOfMeans                 = self.dataHelper.data.groupby([self.labelColumn]).mean(numeric_only=True)
        dataFrameOfMeans["Sample Count"] = self.dataHelper.data[self.labelColumn].value_counts()
        return dataFrameOfMeans


    def GetGroupedByCluster(self):
        """
        Gets a DataFrameGroupBy of the data grouped by each cluster.

        Parameters
        ----------
        None.

        Returns
        -------
        : DataFrameGroupBy
            Grouped clusters.
        """
        return self.dataHelper.data.groupby([self.labelColumn])


    def GetGroupCounts(self):
        """
        Displays the value counts for the specified column as they are grouped the clusterer.

        Parameters
        ----------
        None.

        Returns
        -------
        : Series
            Counts of each cluster.
        """
        return self.dataHelper.data.groupby([self.labelColumn]).sum(numeric_only=False)


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
        PlotHelper.Format()

        figure, axes = plt.subplots(numberOfRows, subPlotColumns)

        figure.set_figwidth(25)
        figure.set_figheight(6*numberOfRows)

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
        PlotHelper.Format()

        if type(columns) != list:
            columns = [columns]

        if not self.labelColumn in columns:
            columns.append(self.labelColumn)

        self.dataHelper.data[columns].groupby(self.labelColumn).mean().plot.bar()
        axes = plt.gca()
        axes.set_title("Feature Mean by Cluster")

        # Turn off the x-axis grid.
        axes.grid(False, axis="x")

        plt.show()