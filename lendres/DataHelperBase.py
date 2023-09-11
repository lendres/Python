"""
Created on July 26, 2022
@author: Lance A. Endres
"""
import pandas                                                   as pd
from   matplotlib                                               import pyplot                     as plt
import seaborn                                                  as sns
from   sklearn.model_selection                                  import train_test_split

from   lendres.ConsoleHelper                                    import ConsoleHelper
from   lendres.plotting.PlotHelper                              import PlotHelper


class DataHelperBase():
    """
    Base class for data helper classes.
    Abstract class, do not instantiate.
    """


    def __init__(self, consoleHelper=None):
        """
        Constructor.

        Parameters
        ----------
        consoleHelper : ConsoleHelper
            Class the prints messages.

        Returns
        -------
        None.
        """
        # Initialize the variables.  Helpful to know if something goes wrong.
        self.data                      = []

        self.xTrainingData             = []
        self.yTrainingData             = []
        self.yTrainingEncoded          = []

        self.xValidationData           = []
        self.yValidationData           = []
        self.yValidationEncoded        = []

        self.xTestingData              = []
        self.yTestingData              = []
        self.yTestingEncoded           = []

        self.labelEncoders             = {}

        # Save the console helper first so it can be used while processing things.
        self.consoleHelper  = None
        if consoleHelper == None:
            self.consoleHelper = ConsoleHelper()
        else:
            self.consoleHelper = consoleHelper


    def Copy(self):
        """
        Creates a copy (copy constructor).

        Parameters
        ----------
        None.

        Returns
        -------
        dataHelper : DataHelperBase subclass
            New DataHelperBase subclass instance with the data of this one copied to it.
        """
        # Creates an instance of the class by calling any subclasses's constructor.
        dataHelper = (type(self))()
        dataHelper.CopyFrom(self)
        return dataHelper


    def CopyFrom(self, dataHelper):
        """
        Copies the data of the DataHelper supplied as input to this DataHelper.

        Parameters
        ----------
        dataHelper : DataHelperBase subclass.
            DataHelper to copy data from.

        Returns
        -------
        None.
        """
        if len(dataHelper.data) != 0:
            self.data                      = dataHelper.data.copy()

        if len(dataHelper.xTrainingData) != 0:
            self.xTrainingData             = dataHelper.xTrainingData.copy()
            self.yTrainingData             = dataHelper.yTrainingData.copy()
            self.yTrainingEncoded          = dataHelper.yTrainingEncoded.copy()

        if len(dataHelper.xValidationData) != 0:
            self.xValidationData           = dataHelper.xValidationData.copy()
            self.yValidationData           = dataHelper.yValidationData.copy()
            self.yValidationEncoded        = dataHelper.yValidationEncoded.copy()

        if len(dataHelper.xTestingData) != 0:
            self.xTestingData              = dataHelper.xTestingData.copy()
            self.yTestingData              = dataHelper.yTestingData.copy()
            self.yTestingEncoded           = dataHelper.yTestingEncoded.copy()

        self.labelEncoders  = dataHelper.labelEncoders.copy()
        self.consoleHelper  = dataHelper.consoleHelper


    def _SplitData(self, x, y, testSize, validationSize=None, stratify=False):
        """
        Splits the data.

        Parameters
        ----------
        x : array like
            Independent data.
        y : array like
            Dependent data.
        testSize : double
            Fraction of the data to use as test data.  Must be in the range of 0-1.
        validationSize : double
            Fraction of the non-test data to use as validation data.  Must be in the range of 0-1.
        stratify : bool
            If true, the approximate ratio of value in the dependent variable is maintained.

        Returns
        -------
        None.
        """
        if len(self.data) == 0:
            raise Exception("Data has not been loaded.")

        if stratify:
            stratifyInput = y
        else:
            stratifyInput = None

        # Split the data.
        self.xTrainingData, self.xTestingData, self.yTrainingData, self.yTestingData = train_test_split(x, y, test_size=testSize, random_state=1, stratify=stratifyInput)

        if validationSize != None:
            if stratify:
                stratifyInput = self.yTrainingData
            else:
                stratifyInput = None
            self.xTrainingData, self.xValidationData, self.yTrainingData, self.yValidationData = train_test_split(self.xTrainingData, self.yTrainingData, test_size=validationSize, random_state=1, stratify=stratifyInput)


    def _GetSplitComparisons(self, originalData, format="countandpercentstring"):
        """
        Returns the value counts and percentages of the dependant variable for the
        original, training (if available), and testing (if available) data.

        Parameters
        ----------
        originalData : array like
            The source data passed from the subclass.
        format : string
            Format of the returned values.
            countandpercentstring  : returns a string that contains both the count and percent.
            numericalcount         : returns the count as a number.
            numericalpercentage    : returns the percentage as a number.

        Returns
        -------
        dataFrame : pandas.DataFrame
            DataFrame with the counts and percentages.
        """
        # Get results for original data.
        dataFrame = self._GetCountAndPrecentStrings(originalData, "Original", format=format)

        # If the data has been split, we will add the split information as well.
        if len(self.yTrainingData) != 0:
            dataFrame = pd.concat([dataFrame, self._GetCountAndPrecentStrings(self.yTrainingData, "Training", format=format)], axis=1)

            if len(self.yValidationData) != 0:
                dataFrame = pd.concat([dataFrame, self._GetCountAndPrecentStrings(self.yValidationData, "Validation", format=format)], axis=1)

            dataFrame = pd.concat([dataFrame, self._GetCountAndPrecentStrings(self.yTestingData, "Testing", format=format)], axis=1)

        return dataFrame


    def _GetCountAndPrecentStrings(self, dataSet, dataSetName, format="countandpercentstring"):
        """
        Calculates the number of each category is in the data set and returns it.  The return format can
        be specified.

        Parameters
        ----------
        dataSet : array like
            Data to extract information from.
        dataSetName : string
            Name of the data set to be used as the column header.
        format : string
            Format of the returned values.
            countandpercentstring  : returns a string that contains both the count and percent.
            numericalcount         : returns the count as a number.
            numericalpercentage    : returns the percentage as a number.

        Returns
        -------
        comparisonFrame : pandas.DataFrame
            DataFrame with the category data.
        """
        valueCounts        = []
        totalCount         = len(dataSet)
        categories         = dataSet.unique()

        formatFunction = None
        if format == "countandpercentstring":
            formatFunction = lambda count, percent : "{0} ({1:0.2f}%)".format(count, percent)
        elif format == "numericalcount":
            formatFunction = lambda count, percent : count
        elif format == "numericalpercentage":
            formatFunction = lambda count, percent : percent
        else:
            raise Exception("Invalid format string specified.")

        # Turn the numbers into formated strings.
        for category in categories:
            classValueCount = sum(dataSet == category)
            valueCounts.append(formatFunction(classValueCount, classValueCount/totalCount*100))

        # Create the data frame.
        comparisonFrame = pd.DataFrame(
            valueCounts,
            columns=[dataSetName],
            index=categories
        )

        return comparisonFrame


    def CreateSplitComparisonPlot(self):
        """
        Plots the split comparisons.

        Parameters
        ----------
        None.

        Returns
        -------
        figure : Matplotlib figure
            The created figure.
        """
        splits  = self.GetSplitComparisons(format="numericalpercentage")
        columns = splits.columns.values
        splits.reset_index(inplace=True)

        PlotHelper.Format()
        axis = splits.plot(x="index", y=columns, kind="bar", color=sns.color_palette())
        axis.set(title="Split Comparison", xlabel="Category", ylabel="Percentage")

        figure = plt.gcf()
        plt.show()

        return figure


    def DisplayDataShapes(self):
        """
        Print out the shape of all the data.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.consoleHelper.PrintTitle("Data Sizes", ConsoleHelper.VERBOSEREQUESTED)
        self.consoleHelper.Display("Data shape:    {0}".format(self.data.shape), ConsoleHelper.VERBOSEREQUESTED)
        self.consoleHelper.Display("Labels length: {0}".format(len(self.data)), ConsoleHelper.VERBOSEREQUESTED)

        if len(self.xTrainingData) != 0:
            self.consoleHelper.PrintNewLine(ConsoleHelper.VERBOSEREQUESTED)
            self.consoleHelper.Display("Training images shape:  {0}".format(self.xTrainingData.shape), ConsoleHelper.VERBOSEREQUESTED)
            self.consoleHelper.Display("Training labels length: {0}".format(len(self.yTrainingData)), ConsoleHelper.VERBOSEREQUESTED)

        if len(self.xValidationData) != 0:
            self.consoleHelper.PrintNewLine(ConsoleHelper.VERBOSEREQUESTED)
            self.consoleHelper.Display("Validation images shape:  {0}".format(self.xValidationData.shape), ConsoleHelper.VERBOSEREQUESTED)
            self.consoleHelper.Display("Validation labels length: {0}".format(len(self.yValidationData)), ConsoleHelper.VERBOSEREQUESTED)

        if len(self.xTestingData) != 0:
            self.consoleHelper.PrintNewLine(ConsoleHelper.VERBOSEREQUESTED)
            self.consoleHelper.Display("Testing images shape:  {0}".format(self.xTestingData.shape), ConsoleHelper.VERBOSEREQUESTED)
            self.consoleHelper.Display("Testing labels length: {0}".format(len(self.yTestingData)), ConsoleHelper.VERBOSEREQUESTED)