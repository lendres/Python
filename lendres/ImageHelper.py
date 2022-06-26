"""
Created on June 24, 2022
@author: Lance A. Endres
"""
from   matplotlib                                import pyplot                     as plt
import pandas                                    as pd
import numpy                                     as np
import tensorflow                                as tf

from   sklearn.model_selection                   import train_test_split

import os
import io

import lendres
from   lendres.ConsoleHelper                     import ConsoleHelper
from   lendres.PlotHelper                        import PlotHelper

class ImageHelper():

    def __init__(self, consoleHelper=None):
        """
        Constructor.

        Parameters
        ----------
        consoleHelper : ConsoleHelper
            Class the prints messages.  The following verbose levels are used.
            Specified how much output should be written. The default is 2.
            0 : None.  Use with caution.
            1 : Erors and warnings are output.
            2 : All messages are output.

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
        self.data            = None
        self.labels          = None
        self.encodedLabels   = None
        self.labelCategories = None


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
        imageHelper                = ImageHelper()
        imageHelper.data           = self.data.copy()
        imageHelper.labels         = self.labels.copy(deep=True)
        imageHelper.consoleHelper  = self.consoleHelper

        if len(self.xTrainingData) != 0:
            imageHelper.xTrainingData             = self.xTrainingData.copy()
            imageHelper.yTrainingData             = self.yTrainingData.copy()

        if len(self.xValidationData) != 0:
            imageHelper.xValidationData           = self.xValidationData.copy()
            imageHelper.yValidationData           = self.yValidationData.copy()

        if len(self.xTestingData) != 0:
            imageHelper.xTestingData              = self.xTestingData.copy()
            imageHelper.yTestingData              = self.yTestingData.copy()

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
        self.labels.rename(columns={self.labels.columns[0] : "Labels"}, inplace=True)

        self.labels["Labels"]   = self.labels["Labels"].astype("category")
        self.labels["Numbers"]  = self.labels["Labels"].cat.codes

        self.labelCategories    = self.labels["Labels"].unique()


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
        self.labelCategories = labelCategories

        self.labels = pd.read_csv(inputFile)
        self.labels.rename(columns={self.labels.columns[0] : "Numbers"}, inplace=True)

        # Initialize label list.
        numberOfLabels = self.labels.shape[0]
        labels = [""] * numberOfLabels

        # Populate labels from selecting them out of the label categories.
        for i in range(numberOfLabels):
            labels[i] = labelCategories[self.labelNumbers[i]]

        self.labels["Labels"] = labels
        self.labels["Labels"] = self.labels["Labels"].astype("category")


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
        if self.data == None:
            raise Exception("Data has not been loaded.")

        if self.encodedLabels == None:
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


    def PlotImage(self, index=0, random=False):
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
        PlotHelper.FormatPlot(width=6, height=6)
        figure = plt.figure()

        # Generating random indices from the data and plotting the images.
        if random:
            index = np.random.randint(0, self.labels.shape[0])

        # Adding subplots with 3 rows and 4 columns.
        axis = plt.gca()

        # Plotting the image using cmap=gray.
        #xis.imshow(self.data[index, :], cmap=plt.get_cmap("gray"))
        axis.imshow(self.data[index, :])
        axis.set_title(self.labels["Labels"].loc[index])

        plt.show()
        PlotHelper.scale = 1.0


    def PlotImages(self, rows=4, columns=4, random=False):
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
        PlotHelper.scale = 0.55
        PlotHelper.FormatPlot(width=columns*2.5+2, height=rows*2.5+2)
        figure = plt.figure()

        numberOfImages = self.labels.shape[0]
        index = -1

        for i in range(columns):
            for j in range(rows):
                # Generating random indices from the data and plotting the images.
                if random:
                    index = np.random.randint(0, numberOfImages)
                else:
                    index = index + 1

                # Adding subplots with 3 rows and 4 columns.
                axis = figure.add_subplot(rows, columns, i*rows+j+1)

                # Plotting the image using cmap=gray.
                #xis.imshow(self.data[index, :], cmap=plt.get_cmap("gray"))
                axis.imshow(self.data[index, :])
                axis.set_title(self.labels["Labels"].loc[index], y=0.9)

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        plt.show()
        PlotHelper.scale = 1.0