"""
Created on July 20, 2023
@author: Lance A. Endres
"""
import pandas                                    as pd
import matplotlib.pyplot                         as plt
import os

from   lendres.algorithms.Search                 import Search
from   lendres.plotting.PlotHelper               import PlotHelper
from   lendres.plotting.AxesHelper               import AxesHelper
from   lendres.plotting.PlotMaker                import PlotMaker

class DataComparison():
    """
    This class is for loading and retrieving data sets that are in different files.

    The data sets should share a common independent axis type (e.g. time), however, they do not have
    to be sampled at the same points.  E.g., one file might sample at 0.01 seconds and the other
    might be sampled at 0.1 seconds.
    """

    def __init__(self, independentColumn:str, directory:str=None):
        """
        Constructor.

        Parameters
        ----------
        independentColumn : str
            The column name of the independent data.
        directory : str, optional
            The directory to load the data files from. The default is None.  If none is supplied,
            the complete path must be specified when loading files.

        Returns
        -------
        None.
        """
        self.independentColumn  = independentColumn
        self.directory          = directory

        self.dataSets           = []
        self.dataSetNames       = []


    @property
    def NumberOfDataSets(self):
        """
        Gets the number of data sets.

        Returns
        -------
        int
        """
        return len(self.dataSets)


    def LoadFile(self, file:str, name:str):
        """
        Loads a data set from file.

        Parameters
        ----------
        file : str
            Path to the file to load.  If a directory was supplied at construction, then the file is
            just the file name (with extension).  Otherwise, it must be the complete path.
        name : str
            The name to give to the data set.

        Returns
        -------
        None.
        """
        dataFrame       = self.ValidateFile(file)
        dataFrame.name  = name
        self.dataSets.append(dataFrame)
        self.dataSetNames.append(name)


    def ValidateFile(self, inputFile:str):
        """
        Validates that a file exists.  Combines the file path with the directory, if one was supplied.

        Parameters
        ----------
        inputFile : str
            File to load.

        Returns
        -------
        : pandas.DataFrame
            The file loaded into a DataFrame.
        """
        path = inputFile
        if self.directory is not None:
            path = os.path.join(self.directory, inputFile)
        if not os.path.exists(path):
            raise Exception("The input file \"" + path + "\" does not exist.")
        return pd.read_csv(path)


    def ValidateData(self):
        dataZero = self.dataSets[0]

        # Make sure starting times are the same.
        for i in range(1, self.NumberOfDataSets):
            data = self.dataSets[i]

            if abs(data[self.independentColumn].iloc[0] - dataZero[self.independentColumn].iloc[0]) != 0:
                raise Exception("The starting times are not equal.")

            # Make sure end times are the same.
            if abs(data[self.independentColumn].iloc[-1] - dataZero[self.independentColumn].iloc[-1]) != 0:
                raise Exception("The ending times are not equal.")


    def GetEndTime(self):
        """
        Get the end time of the data.  Returns the value in the last row of the time column.

        Returns
        -------
        float
            The ending time.
        """
        return (self.dataSets[0])[self.independentColumn].iloc[-1]


    def GetValueAtTime(self, dataSet:int, column:str, time:float):
        """
        Gets the value in the specified column at the specified time from the specified data set.

        Parameters
        ----------
        dataSet : int
            Index of the data set to get the value from.
        column : string
            The name of the column the value is in.
        time : double
            Time of interest to get the velocity.

        Returns
        -------
        value : float
            The value.
        """
        data            = self.dataSets[dataSet]
        boundingIndices = Search.BoundingBinarySearch(time, data[self.independentColumn])
        value = data[column].iloc[boundingIndices[0]]
        return value


    def Apply(self, function):
        """
        Runs a function on every column in the list.

        This runs the "apply" operation on the columns.  Therefore, "function" must take a
        pandas Series as the input.

        Parameters
        ----------
        columns : list of strings
            Columns to operate on.
        function : function
            The function that is applied to each column.

        Returns
        -------
        None.
        """
        for dataSet in self.dataSets:
             function(dataSet)


    def CreateComparisonPlot(self, columns:list, xLabel=None, yLabel:str=None, **kwargs):
        """
        Creates a plot comparing a column from each data set.

        Parameters
        ----------
        columns : list
            The name of the column to compare or a list of column names to compare.
        yLabel : str, optional
            The y-axis label. The default is None.
        **kwargs : keyword arguments
            Keyword arguments to pass to the plot function.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        """
        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        figure = plt.gcf()
        axes   = plt.gca()

        if type(columns) is str:
            columns = [columns]

        i = 0
        for dataSet in self.dataSets:
            for column in columns:
                axes.plot(dataSet[self.independentColumn], dataSet[column], label="Data "+str(i), **kwargs)
                i += 1

        # If no x-axis label is provided, default to the column name.
        if xLabel == None:
            xLabel = self.independentColumn

        # If no y-axis label is provided, default to the column name.
        if yLabel == None:
            yLabel = column

        AxesHelper.Label(axes, title="Comparison of "+column, xLabel=xLabel, yLabels=yLabel)
        axes.grid()

        plt.show()
        return figure


    def CreateMultiAxisComparisonPlot(self, axesesColumnNames:list, yLabels:list, **kwargs):
        """
        Creates a multi y-axes plot.  The columns are plotted for each data set.

        Parameters
        ----------
        axesesColumnNames : array like of array like of strings
            Column names of the data to plot.  The array contains one set (array) of strings for the data to plot on
            each axes.  Example: [[column1, column2], [column3], [column 4, column5]] creates a three axes plot with
            column1 and column2 plotted on the left axes, column3 plotted on the first right axes, and column4 and column5
            plotted on the second right axes.
        yLabels : array like of strings
            A list of strings to use as labels for the y-axes.
        **kwargs : keyword arguments
            Keyword arguments to pass to the plot function.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        """
        figure, axeses = PlotMaker.NewMultiYAxesPlot(self.dataSets, self.independentColumn, axesesColumnNames, colorCycle=None, **kwargs)

        # The AxesHelper can automatically label the axes if you supply it a list of strings for the y labels.
        AxesHelper.Label(axeses, title="Data Comparison", xLabel=self.independentColumn, yLabels=yLabels)

        figure.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=2, bbox_transform=axeses[0].transAxes)
        plt.show()

        return figure