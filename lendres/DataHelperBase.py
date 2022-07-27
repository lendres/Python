"""
Created on July 26, 2022
@author: Lance A. Endres
"""
from   matplotlib                                import pyplot                     as plt
import seaborn                                   as sns

from   lendres.ConsoleHelper                     import ConsoleHelper
from   lendres.PlotHelper                        import PlotHelper


class DataHelperBase():


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
        self.xTrainingData             = []
        self.xValidationData           = []
        self.xTestingData              = []

        self.yTrainingData             = []
        self.yValidationData           = []
        self.yTestingData              = []

        self.data                      = []

        self.labelEncoders             = {}

        # Save the console helper first so it can be used while processing things.
        self.consoleHelper  = None
        if consoleHelper == None:
            self.consoleHelper = ConsoleHelper()
        else:
            self.consoleHelper = consoleHelper


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

        PlotHelper.FormatPlot()
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