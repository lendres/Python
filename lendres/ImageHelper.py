"""
Created on June 24, 2022
@author: Lance A. Endres
"""
from   matplotlib                                import pyplot                     as plt
import seaborn                                   as sns
import pandas                                    as pd
import numpy                                     as np
import tensorflow                                as tf
import cv2

import random

from   sklearn.model_selection                   import train_test_split

import lendres
from   lendres.ConsoleHelper                     import ConsoleHelper
from   lendres.PlotHelper                        import PlotHelper
from   lendres.UnivariateAnalysis                import UnivariateAnalysis
from   lendres.Algorithms                        import FindIndicesByValues

class ImageHelper():
    arrayImageSize = 2.5

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
        self.xTrainingData             = []
        self.xTestingData              = []
        self.xValidationData           = []

        self.yTrainingData             = []
        self.yValidationData           = []
        self.yTestingData              = []

        # Save the console helper first so it can be used while processing things.
        self.consoleHelper  = None
        if consoleHelper == None:
            self.consoleHelper = ConsoleHelper()
        else:
            self.consoleHelper = consoleHelper

        # Initialize the variable.  Helpful to know if something goes wrong.
        self.data                    = []
        self.labels                  = []
        self.encodedLabels           = []
        self.labelCategories         = []
        self.numberOfLabelCategories = 0
        self.colorConversion         = None


    def Copy(self):
        """
        Creates a copy (copy constructor).

        Parameters
        ----------
        deep : bool, optional
            Specifies if a deep copy should be done. The default is False.

        Returns
        -------
        None.
        """
        imageHelper                         = ImageHelper()
        imageHelper.data                    = self.data.copy()
        imageHelper.labels                  = self.labels.copy(deep=True)
        imageHelper.consoleHelper           = self.consoleHelper

        imageHelper.xTrainingData           = self.xTrainingData.copy()
        imageHelper.yTrainingData           = self.yTrainingData.copy()

        imageHelper.xValidationData         = self.xValidationData.copy()
        imageHelper.yValidationData         = self.yValidationData.copy()

        imageHelper.xTestingData            = self.xTestingData.copy()
        imageHelper.yTestingData            = self.yTestingData.copy()

        imageHelper.encodedLabels           = self.encodedLabels.copy()
        imageHelper.labelCategories         = self.labelCategories.copy()
        imageHelper.numberOfLabelCategories = self.numberOfLabelCategories

        imageHelper.colorConversion         = self.colorConversion

        return imageHelper


    def LoadImagesFromNumpyArray(self, inputFile):
        """
        Loads a data file of images stored as a numpy array.

        Parameters
        ----------
        inputFile : string
            Path and name of the file to load.

        Returns
        -------
        None.
        """
        self.data = np.load(inputFile)


    def LoadLabelsFromCsv(self, inputFile):
        """
        Loads the labels from a CSV file.

        Parameters
        ----------
        inputFile : string
            Path and name of the file to load.

        Returns
        -------
        None.
        """
        self.labels = pd.read_csv(inputFile)
        self.labels.rename(columns={self.labels.columns[0] : "Names"}, inplace=True)

        self.labels["Names"]   = self.labels["Names"].astype("category")
        self.labels["Numbers"] = self.labels["Names"].cat.codes

        uniqueLabels = self.labels["Names"].unique().categories.tolist()
        self.SetLabelCategories(uniqueLabels)


    def LoadLabelNumbersFromCsv(self, inputFile, labelCategories):
        """
        Loads the labels from a CSV file.

        Parameters
        ----------
        inputFile : string
            Path and name of the file to load.

        Returns
        -------
        None.
        """
        self.SetLabelCategories(labelCategories)

        self.labels = pd.read_csv(inputFile)
        self.labels.rename(columns={self.labels.columns[0] : "Numbers"}, inplace=True)

        # Initialize label list.
        numberOfLabels = self.labels.shape[0]
        labels = [""] * numberOfLabels

        # Populate labels from selecting them out of the label categories.
        for i in range(numberOfLabels):
            labels[i] = labelCategories[self.labelNumbers[i]]

        self.labels["Names"] = labels
        self.labels["Names"] = self.labels["Names"].astype("category")


    def SetLabelCategories(self, labelCategories):
        """
        Sets the category labels.  These are the unique text labels of the dependent variable.  That is, while
        the text labels and numerical labels for the dependent variable are the length of the number of data samples,
        this is only a few text labels as only one of each category is present.

        The labels should be in the same order as the numerical labels such that a numerical label of i returns the correct
        text name of the category by using self.labelCategories[i].

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.labelCategories         = labelCategories
        self.numberOfLabelCategories = len(labelCategories)


    def DisplayLabelCategories(self):
        """
        Neatly displays the label categories.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        labelsDataFrame = pd.DataFrame(self.labelCategories, index=range(0, self.numberOfLabelCategories), columns=["Labels"])
        self.consoleHelper.Display(labelsDataFrame)


    def NormalizePixelValues(self):
        """
        Normalizes all the pixel valus.  Call before splitting the data.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.data = self.data / 255


    def EncodeCategoricalColumns(self):
        """
        Converts the categorical columns ("category" data type) to encoded values.
        Prepares categorical columns for use in a model.

        Parameters
        ----------
        columns : list of strings
            The names of the columns to encode.
        dropFirst : bool
            If true, the first category is dropped for the encoding.

        Returns
        -------
        None.
        """
        self.encodedLabels = tf.keras.utils.to_categorical(self.labels["Numbers"])


    def DisplayEncodingResults(self, numberOfEntries, randomEntries=False):
        """
        Prints a summary of the encoding processes.

        Parameters
        ----------
        numberOfEntries : int
            The number of entries to display.
        randomEntries : bool
            If true, random entries are chosen, otherwise, the first few entries are displayed.

        Returns
        -------
        None.
        """
        indices = []
        if randomEntries:
            numberOfImages = self.labels.shape[0]
            indices = random.sample(range(0, numberOfImages), numberOfEntries)
        else:
            indices = list(range(numberOfEntries))

        self.consoleHelper.PrintTitle("Dependent Variable Numerical Labels")
        self.consoleHelper.Display(pd.DataFrame(self.labels["Numbers"].loc[indices]));

        self.consoleHelper.PrintNewLine()
        self.consoleHelper.PrintTitle("Dependent Variable Text Labels")
        self.consoleHelper.Display(pd.DataFrame(self.labels["Names"].loc[indices]));

        self.consoleHelper.PrintNewLine()
        self.consoleHelper.PrintTitle("Dependent Variable Encoded Labels")
        self.consoleHelper.Display(pd.DataFrame(self.encodedLabels[indices]));


    def SplitData(self, testSize, validationSize=None, stratify=False):
        """
        Creates a linear regression model.  Splits the data and creates the model.

        Parameters
        ----------
        dependentVariable : string
            Name of the column that has the dependant data.
        testSize : double
            Fraction of the data to use as test data.  Must be in the range of 0-1.
        validationSize : double
            Fraction of the non-test data to use as validation data.  Must be in the range of 0-1.
        stratify : bool
            If true, the approximate ratio of value in the dependent variable is maintained.

        Returns
        -------
        data : pandas.DataFrame
            Data in a pandas.DataFrame
        """
        if len(self.data) == 0:
            raise Exception("Data has not been loaded.")

        if len(self.encodedLabels) == 0:
            raise Exception("Dependent variable not encoded.")

        x = self.data
        y = self.encodedLabels

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

    def GetSplitComparisons(self):
        """
        Returns the value counts and percentages of the dependant variable for the
        original, training (if available), and testing (if available) data.

        Parameters
        ----------
        None.

        Returns
        -------
        comparisonFrame : pandas.DataFrame
            DataFrame with the counts and percentages.
        """
        # Get results for original data.
        dataFrame = self.GetCountAndPrecentStrings(self.encodedLabels ,"Original")

        # If the data has been split, we will add the split information as well.
        if len(self.yTrainingData) != 0:
            dataFrame = pd.concat([dataFrame, self.GetCountAndPrecentStrings(self.yTrainingData, "Training")], axis=1)

            if len(self.yValidationData) != 0:
                dataFrame = pd.concat([dataFrame, self.GetCountAndPrecentStrings(self.yValidationData, "Validation")], axis=1)

            dataFrame = pd.concat([dataFrame, self.GetCountAndPrecentStrings(self.yTestingData, "Testing")], axis=1)

        return dataFrame


    def GetCountAndPrecentStrings(self, dataSet, dataSetName):
        """
        Gets a string that is the value count of "classValue" and the percentage of the total
        that the "classValue" accounts for in the column.

        Parameters
        ----------
        dataSet : string
            Which data set(s) to plot.

        Returns
        -------
        None.
        """
        numberOfCategories = len(self.labelCategories)
        valueCounts        = [""] * numberOfCategories
        totalCount         = dataSet.shape[0]

        # This counts all the ones for each column.
        searchList         = [1] * numberOfCategories
        classValueCount    = sum(dataSet == searchList)

        # Turn the numbers into formated strings.
        for i in range(numberOfCategories):
            valueCounts[i] = "{0} ({1:0.2f}%)".format(classValueCount[i], classValueCount[i]/totalCount*100)

        # Create the data frame.
        comparisonFrame = pd.DataFrame(
            valueCounts,
            columns=[dataSetName],
            index=self.labelCategories
        )

        return comparisonFrame


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
        self.consoleHelper.PrintTitle("Data Sizes")
        self.consoleHelper.Display("Data shape:    {0}".format(self.data.shape))
        self.consoleHelper.Display("Labels length: {0}".format(len(self.data)))

        if len(self.xTrainingData) != 0:
            self.consoleHelper.PrintNewLine()
            self.consoleHelper.Display("Training images shape:  {0}".format(self.xTrainingData.shape))
            self.consoleHelper.Display("Training labels length: {0}".format(len(self.yTrainingData)))

        if len(self.xValidationData) != 0:
            self.consoleHelper.PrintNewLine()
            self.consoleHelper.Display("Validation images shape:  {0}".format(self.xValidationData.shape))
            self.consoleHelper.Display("Validation labels length: {0}".format(len(self.yValidationData)))

        if len(self.xTestingData) != 0:
            self.consoleHelper.PrintNewLine()
            self.consoleHelper.Display("Testing images shape:  {0}".format(self.xTestingData.shape))
            self.consoleHelper.Display("Testing labels length: {0}".format(len(self.yTestingData)))


    def PlotImage(self, index=0, random=False, size=6):
        """
        Print example images.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # Defining the figure size.  Automatically adjust for the number of images to be displayed.
        #PlotHelper.scale = 0.65
        PlotHelper.FormatPlot(width=size, height=size)

        # Generating random indices from the data and plotting the images.
        if random:
            index = np.random.randint(0, self.labels.shape[0])

        # Adding subplots with 3 rows and 4 columns.
        axis = plt.gca()

        # Plotting the image.
        image = self.data[index]
        if self.colorConversion != None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axis.imshow(image)

        axis.set_title(self.labels["Names"].loc[index])

        # Turn off white grid lines.
        plt.grid(False)

        plt.show()
        PlotHelper.scale = 1.0


    def PlotImages(self, rows=4, columns=4, random=False, indices=None):
        """
        Plot example images.

        Parameters
        ----------
        rows : integer
            The number of rows to plot.
        columns : integer
            The number of columns to plot.
        random : boolean
            If true, random images are selected and plotted.  This overrides the value of indices.  That is,
            if true, random images will plot regardless of the value of indices (None or provided).
        indices : list of integers
            If provided, these images are plotted.  If not provided, the first rows*columns are plotted.  Note,
            this argument only has an effect if random=False.

        Returns
        -------
        None.
        """
        # Defining the figure size.  Automatically adjust for the number of images to be displayed.
        PlotHelper.scale = 0.55
        PlotHelper.FormatPlot(width=columns*ImageHelper.arrayImageSize+2, height=rows*ImageHelper.arrayImageSize+2)
        figure = plt.figure()

        numberOfImages = self.labels.shape[0]

        # If no indices are provided, we are going to print out the start of the data.
        if indices == None:
            indices = range(rows*columns)

        # Position in the index array/range.
        k = -1

        for i in range(columns):
            for j in range(rows):
                # Generating random indices from the data and plotting the images.
                if random:
                    index = np.random.randint(0, numberOfImages)
                else:
                    k += 1
                    index = indices[k]

                # Adding subplots with 3 rows and 4 columns.
                axis = figure.add_subplot(rows, columns, i*rows+j+1)

                # Plotting the image using cmap=gray.
                image = self.data[index]
                if self.colorConversion != None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                axis.imshow(image)

                # Turn off white grid lines.
                axis.grid(False)

                axis.set_title(self.labels["Names"].loc[index], y=0.9)

        # Adjust spacing so titles don't run together.
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

        plt.show()
        PlotHelper.scale = 1.0


    def PlotImageExamplesByCategory(self, numberOfExamples, categoryName=None, categoryNumber=None):
        """
        Plots example images of the specified category type.

        Parameters
        ----------
        numberOfExamples : integer
            The number of examples of the category to plot.
        categoryName : string
            The name of the category to plot examples of.  Provide the name or the number, but not both.
        categoryNumber : integory
            The number of the category to plot examples of.  Provide the name or the number, but not both.

        Returns
        -------
        None.
        """
        # If a name was provided, convert it to a number.  Searching by number is faster.
        if categoryName != None:
            categoryNumber = FindIndicesByValues(self.labelCategories, searchValue=categoryName, maxCount=1)
            # The find indices function returns an array, we only want one entry.
            categoryNumber = categoryNumber[0]

        # If neither name nor number is provided, it is an error.
        if categoryNumber == None:
            raise Exception("A valid category name or a valid category number must be provided.")

        indices = FindIndicesByValues(self.labels["Numbers"], categoryNumber, numberOfExamples)

        if len(indices) == 0:
            categoryName = self.labelCategories[categoryNumber]
            raise Exception("No examples of the category were found.\nCategory Number: "+str(categoryNumber)+"\nCategory Name: "+categoryName)

        self.PlotImages(1, numberOfExamples, indices=indices)


    def PlotImageExamplesForAllCategories(self, numberOfExamples):
        """
        Plots example images of each category type.

        Parameters
        ----------
        numberOfExamples : integer
            The number of examples of each category to plot.

        Returns
        -------
        None.
        """
        # Note, the below assumes all categories are numbered in sequence from 0 to numberOfLabelCategories.  That is, the
        # categories cannot be arbitrarily numbered.
        for i in range(self.numberOfLabelCategories):
            # For subsequent sections, add space after the preceeding.
            if i > 0:
                self.consoleHelper.PrintNewLine()

            self.consoleHelper.PrintTitle(self.labelCategories[i])
            self.PlotImageExamplesByCategory(numberOfExamples, categoryNumber=i)


    def CreateCountPlot(self):
        """
        Creates a count plot of the image categories.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        PlotHelper.FormatPlot()
        axis = sns.countplot(x=self.labels["Names"], palette=sns.color_palette("mako"))
        axis.set(title="Category Count Plot", xlabel="Category", ylabel="Count")
        UnivariateAnalysis.LabelPercentagesOnCountPlot(axis)

        # The categories are labeled by name (text) so rotate the text so it does not run together.
        axis.set_xticklabels(axis.get_xticklabels(), rotation=45, ha="right")
        plt.show()

    def ApplyGaussianBlur(self, **kwargs):
        newImages = []

        print(self.data.shape[0])

        for i in range(self.data.shape[0]):
            newImages.append(cv2.GaussianBlur(self.data[i], **kwargs))

        self.data = newImages