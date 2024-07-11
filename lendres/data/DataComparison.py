"""
Created on July 20, 2023
@author: Lance A. Endres
"""
import pandas                                                        as pd
import matplotlib.pyplot                                             as plt
import os

from   lendres.algorithms.Search                                     import Search
from   lendres.plotting.PlotHelper                                   import PlotHelper
from   lendres.plotting.AxesHelper                                   import AxesHelper
from   lendres.plotting.PlotMaker                                    import PlotMaker
from   lendres.plotting.LegendHelper                                 import LegendHelper
from   lendres.plotting.LegendOptions                                import LegendOptions

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
        self.AddDataSet(dataFrame, name)


    def AddDataSet(self, dataFrame:pd.DataFrame, name:str):
        """
        Add a data set from an existing DataFrame.

        Parameters
        ----------
        dataFrame : pd.DataFrame
            A data set as a pandas.DataFrame.
        name : str
            The name to give to the data set.

        Returns
        -------
        None.
        """
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


    def GetEndTime(self):
        """
        Get the end time of the data.  Returns the value in the last row of the time column.

        Returns
        -------
        float
            The ending time.
        """
        return (self.dataSets[0])[self.independentColumn].iloc[-1]


    def GetValue(self, dataSet:int, column:str, time:float):
        """
        Gets the value in the specified column at the specified value of the independent axis.  The value is returned
        from the specified data set.

        Parameters
        ----------
        dataSet : int
            Index of the data set to get the value from.
        column : string
            The name of the column the value is in.
        time : double
            Time of interest.

        Returns
        -------
        value : float
            The value.
        """
        index = self.GetIndex(dataSet, time)
        data  = self.dataSets[dataSet]
        value = data[column].iloc[index]
        return value


    def GetIndex(self, dataSet:int, time:float):
        """
        Gets the index at the specified value of the independent axis.  The index is returned from the specified data set.

        Parameters
        ----------
        dataSet : int
            Index of the data set to get the value from.
        time : double
            Time of interest.

        Returns
        -------
        : int
            The index (or closest index if the time value does not exist) to the specified time.
        """
        data            = self.dataSets[dataSet]
        boundingIndices = Search.BoundingBinarySearch(time, data[self.independentColumn])
        return boundingIndices[0]


    def Apply(self, function):
        """
        Runs a function on every data set.

        Parameters
        ----------
        function : function
            The function that is applied to each data set.  The function should take a pandas.DataFrame as the input.

        Returns
        -------
        None.
        """
        for dataSet in self.dataSets:
             function(dataSet)


    def CreateComparisonPlot(self, columns:list, title:str=None, xLabel:str=None, yLabel:str=None, legendOptions:LegendOptions=LegendOptions(), **kwargs):
        figure, axes = self.NewComparisonPlot(columns, title, xLabel, yLabel, legendOptions, **kwargs)
        LegendHelper.CreateLegendAtFigureBottom(figure, axes, offset=0.15*PlotHelper.GetSettings().Scale, legendOptions=legendOptions)
        plt.show()
        return figure


    def NewComparisonPlot(
            self, columns:list,
            title:str=None,
            xLabel:str=None,
            yLabel:str|list=None,
            labelSuffixes:str=None,
            **kwargs
        ):
        """
        Creates a plot comparing a column from each data set.

        Parameters
        ----------
        columns : list
            The name of the column to compare or a list of column names to compare.
        title : str, optional
            The plot title. The default is None.
        xLabel : str, optional
            The x-axis label. The default is None.
        yLabel : str, optional
            The y-axis label. The default is None.
        labelSuffixes : str, optional
            The label suffix to append for each series plotted.  If None, then the column name is used.  If supplied, the number of
            of values supplied must equal len(columns).  The default is None.
        **kwargs : keyword arguments
            Keyword arguments to pass to the plot function.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        """
        # Must be run before creating figure or plotting data.
        PlotHelper.Format()

        figure = plt.gcf()
        axes   = plt.gca()

        if type(columns) is str:
            columns = [columns]

        if type(labelSuffixes) is str:
            labelSuffixes = [labelSuffixes]

        if labelSuffixes is None:
            labelSuffixes = columns

        # Convert the kwargs into individual series kwargs.
        seriesKeyWordArgs = PlotHelper.ConvertKeyWordArgumentsToSeriesSets(len(columns)*len(self.dataSets), **kwargs)

        i = 0
        for dataSet, dataSetName in zip(self.dataSets, self.dataSetNames):
            for column, labelSuffix in zip(columns, labelSuffixes):
                label = dataSetName + " " + labelSuffix
                axes.plot(dataSet[self.independentColumn], dataSet[column], label=label, **(seriesKeyWordArgs[i]))
                i += 1

        # If no title is provided, create a default.
        if title is None:
            title = "Comparison of "+column

        # If no x-axis label is provided, default to the column name.
        if xLabel is None:
            xLabel = self.independentColumn

        # If no y-axis label is provided, default to the column name.
        if yLabel is None:
            yLabel = column

        AxesHelper.Label(axes, title=title, xLabels=xLabel, yLabels=yLabel)

        return figure, axes


    def CreateMultiAxisComparisonPlot(self, axesesColumnNames:list, yLabels:list, legendOptions:LegendOptions=LegendOptions(), **kwargs):
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
        legendOptions : LegendOptions, optional
            Options that specify if and how the legend is generated. The default is LegendOptions().
        **kwargs : keyword arguments
            Keyword arguments to pass to the plot function.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        """
        figure, axeses = PlotHelper.NewMultiYAxesFigure(len(axesesColumnNames))

        for dataSet in self.dataSets:
            lines = PlotMaker.MultiAxesPlot(axeses, dataSet, self.independentColumn, axesesColumnNames, independentAxis="x", **kwargs)
            for line in lines:
                line.set_label(dataSet.name + " " + line.get_label())

        AxesHelper.AlignYAxes(axeses)

        # The AxesHelper can automatically label the axes if you supply it a list of strings for the y labels.
        AxesHelper.Label(axeses, title="Data Comparison", xLabels=self.independentColumn, yLabels=yLabels)

        figure.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=2, bbox_transform=axeses[0].transAxes)
        plt.show()

        return figure