# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 18:06:31 2021

@author: Lance
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
from IPython.display import display
import os
import io

import lendres
from lendres.ConsoleHelper import ConsoleHelper

class DataHelper:

    def __init__(self, fileName=None, data=None, copy=False, deep=False, consoleHelper=None):
        """
        Constructor.

        Parameters
        ----------
        fileName : stirng, optional
            Path to load the data from.  This is a shortcut for creating a DataHelper and
            then calling "LoadAndInspectData."
        data : pandas.self.dataFrame, optional
            DataFrame to operate on. The default is None.  If None is specified, the
            data should be loaded in a separate function call, e.g., with "LoadAndInspectData"
            or by providing a fileName to load the data from.  You cannot provide both a file
            and data.
        deep : bool, optional
            Specifies if a deep copy should be done. The default is False.  Only valid if
            the "self.data" parameter is specified.
        consolueHelper : ConsolueHelper
            Class the prints messages.  The following verbose levels are used.
            Specified how much output should be written. The default is 2.
            0 : None.  Use with caution.
            1 : Erors and warnings are output.
            2 : All messages are output.

        Returns
        -------
        None.

        """
        # Save the console helper first so it can be used while processing things.
        self.consoleHelper  = None
        if consoleHelper == None:
            self.consoleHelper = ConsoleHelper()
        else:
            self.consoleHelper = consoleHelper

        # Initialize the variable.  Helpful to know if something goes wrong.
        self.data = None

        # Either load the data from file or the supplied existing data, but not both.
        if fileName != None:
            self.LoadAndInspectData(fileName)

        elif data != None:
            if copy:
                self.data = self.data.copy(deep)
            else:
                self.data = self.data


    @classmethod
    def Copy(cls, original, deep=False):
        """
        Creates a copy from another DataHelper (copy constructor).

        Parameters
        ----------
        original : DataHelper
            The source of the data.
        deep : bool, optional
            Specifies if a deep copy should be done. The default is False.

        Returns
        -------
        None.

        """
        dataHelper = DataHelper()
        dataHelper.data = original.data.copy(deep)
        return dataHelper


    def LoadAndInspectData(self, inputFile):
        """
        Loads a self.data file and performs some initial inspections and reports results.

        Parameters
        ----------
        inputFile : string
            Path and name of the file to load.

        Returns
        -------
        self.data : pandas.self.dataFrame
            self.data in a pandas.self.dataFrame
        """

        # Validate the input file.
        if type(inputFile) != str:
            raise Exception("The input file is not a string.")

        if not os.path.exists(inputFile):
            raise Exception("The input file does not exist.")


        # Read the file in.
        self.consoleHelper.Print("\nInput file: "+"\n"+inputFile)
        self.data = pd.read_csv(inputFile)

        self.consoleHelper.Print("\nData size:")
        self.consoleHelper.Display(self.data.shape)


        self.consoleHelper.Print("\nFirst few records:")
        self.consoleHelper.Display(self.data.head())

        np.random.seed(1)
        self.consoleHelper.Print("\nRecord random sampling:")
        self.consoleHelper.Display(self.data.sample(n=10))

        self.consoleHelper.Print("\nData summary:")
        self.consoleHelper.Display(self.data.describe())

        # Check self.data types.
        buffer = io.StringIO()
        self.data.info(buf=buffer)
        self.consoleHelper.Print("\nData types:")
        self.consoleHelper.Print(buffer.getvalue())

        # Check unique value counts.
        self.consoleHelper.Print("\nUnique counts:")
        self.consoleHelper.Display(self.data.nunique())

        # See if there are any missing entries, if so they will have to be cleaned.
        self.PrintNotAvailableCounts()

        return self.data


    def PrintNotAvailableCounts(self):
        """
        Prints the counts of any missing (not available) entries.

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.data in a pandas.self.dataFrame

        Returns
        -------
        None.
        """
        notAvailableCounts, totalNotAvailable = self.GetNotAvailableCounts()

        self.consoleHelper.Print("\nMissing entry counts:")
        self.consoleHelper.Display(notAvailableCounts)

        if totalNotAvailable:
            self.consoleHelper.PrintWarning("Some data entries are missing.")
            self.consoleHelper.Print("Total missing: "+str(totalNotAvailable))
        else:
            self.consoleHelper.Print("No entries are missing.")


    def GetNotAvailableCounts(self):
        """
        Gets the counts of any missing (not available) entries.

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.data in a pandas.self.dataFrame

        Returns
        -------
        None.
        """
        notAvailableCounts = self.data.isna().sum()
        totalNotAvailable = sum(notAvailableCounts)
        return notAvailableCounts, totalNotAvailable


    def ChangeToCategoryType(self, categories):
        """
        Changes the self.data series specified to type "category."

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.dataFrame to change the categories in.
        categories : list, array of strings
            Names of the categories to change to the category self.data type.

        Returns
        -------
        None.
        """

        for category in categories:
            self.data[category] = self.data[category].astype("category")


    def DropRowsWhereDataNotAvailable(self, category, inPlace=False):
        """
        Drops any rows that do not have self.data available in the category of "category."

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.dataFrame to change the categories in.
        category : string
            Names of the category to look for not available entries.
        inPlace : bool
            If true, the modifications are done in place.

        Returns
        -------
        : pandas.self.dataFrame or None
            self.dataFrame without the removed rows or None if inPlace=True.
        """
        # Gets an self.dataSeries of boolean values indicating where values were not available.
        indexMask = self.data[category].isna()

        # The indexMask is a self.dataSeries that has the indexes from the original self.dataFrame and the values are the result
        # of the test statement (bools).  The indices do not necessarily correspond to the location in the self.dataFrame.  For
        # example some rows may have been removed.  Extract only the indices we want to remove by using the mask itself.
        dropIndices = indexMask[indexMask].index

        # Drop the rows.
        return self.data.drop(dropIndices, inplace=inPlace)


    def DropAllRowsWhereDataNotAvailable(self, inPlace=False):
        """
        Drops any rows that are missing one or more entries from the self.data.

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.dataFrame to change the categories in.
        inPlace : bool
            If true, the modifications are done in place.

        Returns
        -------
        : pandas.self.dataFrame or None
            self.dataFrame without the removed rows or None if inPlace=True.
        """
        # Gets an self.dataSeries of boolean values indicating where values were not available.
        # to_numpy returns array inside of a tuple for some odd reason.  The [0] extracts the array.
        dropIndices = self.data.isna().sum(axis=1).to_numpy().nonzero()[0]

        # Drop the rows.
        return self.data.drop(dropIndices, inplace=inPlace)


    def GetRowsWithMissingEntries(self):
        """
        Gets the rows that contain missing self.data.

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.dataFrame to change the categories in.

        Returns
        -------
        locations : numpy array
            A list that contains the indices of the rows that contain at least one missing entry.
        count : int
            Number of rows with missing entries.
        """

        locations = self.data.isna().sum(axis=1).to_numpy().nonzero()[0]
        count     = len(locations)

        return locations, count


    def ExtractLastStringTokens(self, categories):
        """
        Gets last string token from all the categories.

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.dataFrame to change the categories in.
        categories : list, array of strings
            Names of the categories to operate on.

        Returns
        -------
        self.dataFrame : pandas.self.dataFrame
            A self.dataFrame that contains the indices of the rows that contain at least one missing entry.
        """
        # Initialize variables.
        numberOfRows  = self.data.shape[0]
        self.dataFrame     = pd.self.dataFrame(index=range(numberOfRows), columns=categories)

        # Extract all the units information from the cells.  Loop over all the cells and extract the second half of the split string.
        for category in categories:
            for i in range(numberOfRows):
                value = self.data[category].iloc[i]
                self.dataFrame[category].iloc[i] = value.split()[1]

        return self.dataFrame


    def KeepFirstStringToken(self, value):
        """
        Takes a string that has multiple tokens and keeps only the first.

        Parameters
        ----------
        value : string
            The self.data entry.

        Returns
        -------
        : string
            The first token of the string.
        """

        # Make sure we are processing a string.
        if isinstance(value, str):
            # Splits the string at the space and returns the first entry as a number.
            return value.split()[0]

        else:
            # Entry wasn't a string, return an empty string.
            return ""


    def KeepFirstTokenAsNumber(self, value):
        """
        Takes a string that has two tokens (a number followed by a string), extracts the numerical part, and returns it as a float.

        Parameters
        ----------
        value : string
            The self.data entry.

        Returns
        -------
        value: float
            The first token of the string as a number of np.nan if the entry was not a string or number.
        """

        # Make sure we are processing a string.
        if isinstance(value, str):
            # Splits the string at the space and returns the first entry as a number.
            return float(value.split()[0])

        elif isinstance(value, float):
            # Already a number, return it.
            return value

        else:
            # Entry wasn't a string or number, so return an out of range value.
            return np.nan


    def ConvertMileage(self, value):
        """
        Takes a value from the mileage category, strips the units string, converts the units, and returns it as a float.

        Parameters
        ----------
        value : string
            The self.data entry.

        Returns
        -------
        value : float
            The mileage as km per liter or np.nan if entry was not a string or number.
        """

        # Make sure we are processing a string.
        if isinstance(value, str):
            # Splits the string at the space and returns the first entry as a number.
            splitString = value.split()
            value       = float(splitString[0])

            if splitString[1] == "km/kg":
                # Approximately 1.35 kg/liter.
                # km   1.35 kg   km
                # -- * ------  = --
                # kg     l       l
                value *= 1.35

            return value

        elif isinstance(value, float):
            # Already a number, return it.
            return value

        else:
            # Entry wasn't a string or number, so return an out of range value.
            return np.nan


    def GetMinAndMaxValues(self, category, criteria, method="quantity"):
        """
        Display and maximum and minimum values in a self.dataSeries.

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.dataFrame to change the categories in.
        category : string
            Names of the category to sort and display.
        criteria : list of two floats or a float
            The criteria used to determine which numbers are dropped.  See "method" for more information.
        method : string
            How to determine which values are dropped.
                percent - A percentage of the top and bottom values are dropped.
                quantity - The number of values specified is dropped.

        Returns
        -------
        : pandas.self.dataFrame
            A self.dataFrame that has both the minimum and maximum values, along with the indices where those values
            occur.  The self.dataFrame contains the following headings:
            Smallest_Index", "Smallest", "Largest_Index", "Largest"
        """
        # Initialize the variable so it is in scope.
        numberOfRows = criteria

        if method == "quantity":
            # Handled by the initialization above, no need to do anything except that the stupid ass Python parser
            # thinks it needs something.
            numberOfRows = criteria
        elif method == "percent":
            # Convert the fraction to a number of rows.
            numberOfRows = round(len(self.data[category]) * criteria / 100)
        else:
            # A boo-boo was made.
            raise Exception("Invalid \"method\" specified.")

        # Sort then display the start and end of the series.
        sortedSeries = self.data[category].sort_values()

        # Create new self.dataFrames for the head (smallest values) and the tail (largest values).
        # Reset the index to move the index to a column and create a new, renumbered index.  This lets us combine the two self.dataFrames at
        # the same index and saves the indices so we can use them later.
        # Also rename the columns to make them more meaningful.
        head = sortedSeries.head(numberOfRows).reset_index()
        head.rename({"index" : "Smallest_Index", category : "Smallest"}, axis=1, inplace=True)

        tail = sortedSeries.tail(numberOfRows).reset_index()
        tail.rename({"index" : "Largest_Index", category : "Largest"}, axis=1, inplace=True)

        # Combine the two along the columns and return the result.
        return pd.concat([head, tail], axis=1)


    def ReplaceLowOutlierWithMean(self, category, criteria):
        """
        Replaces values beyond the criteria with the mean value for the category.  Done in place.

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            The self.data.
        category: string
            Category name in the self.dataFrame.

        Returns
        -------
        None.
        """
        mean = self.data[category].mean()
        self.data.loc[self.data[category] < criteria, category] = mean


    def DropMinAndMaxValues(self, category, criteria, method="fraction", inPlace=False):
        """
        Drops any rows that do not have self.data available in the category of "category."

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.dataFrame to change the categories in.
        category : string
            Names of the category to look for not available entries.
        criteria : list of two floats or a float
            The criteria used to determine which numbers are dropped.  See "method" for more information.
        method : string
            How to determine which values are dropped.
                fraction - A percentage of the top and bottom values are dropped.
                quantity - The number of values specified is dropped.
        inPlace : bool
            If true, the modifications are done in place.

        Returns
        -------
        self.data : pandas.self.dataFrame
            self.dataFrame without the removed rows or None if inPlace=True.
        """

        # This will return a struction that has the values and the indices of the minimums and maximums.
        minAndMaxValues = self.GetMinAndMaxValues(category, criteria, method=method)

        # numpy.where returns array inside of a tuple for some odd reason.  The [0] extracts the array.
        dropIndices = pd.concat([minAndMaxValues["Smallest_Index"], minAndMaxValues["Largest_Index"]])

        # Drop the rows.
        return self.data.drop(dropIndices, inplace=inPlace)


    def DropOutliers(self, column, irqScale=1.5, inPlace=False):
        """
        Drops any rows that are considered outliers by the definition of

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.dataFrame to change the categories in.
        column : string
            Names of the column to look for not available entries.
        irqScale : float
            Scale factor of interquartile range used to define outliers.
        inPlace : bool
            If true, the modifications are done in place.

        Returns
        -------
        self.data : pandas.self.dataFrame
            self.dataFrame without the removed rows or None if inPlace=True.
        """

        # Get the stats we need.
        interQuartileRange = stats.iqr(self.data[column])
        limits             = np.quantile(self.data[column], q=(0.25, 0.75))

        # Set the outlier limits.
        limits[0] -= irqScale*interQuartileRange
        limits[1] += irqScale*interQuartileRange

        # Gets an self.dataSeries of boolean values indicating where values are outside of the range.  These are the
        # values we want to drop.
        indexMask = (self.data[column] < limits[0]) | (self.data[column] > limits[1])

        # The indexMask is a self.dataSeries that has the indexes from the original self.dataFrame and the values are the result
        # of the test statement (bools).  The indices do not necessarily correspond to the location in the self.dataFrame.  For
        # example some rows may have been removed.  Extract only the indices we want to remove by using the mask itself.
        dropIndices = indexMask[indexMask].index

        # Drop the rows.
        return self.data.drop(dropIndices, inplace=inPlace)


    def RemoveAllUnusedCategories(self):
        """
        Removes any unused categories (value count is zero) from all series that are of type "category."
        Performs operation "in place."

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.dataFrame to operate on.

        Returns
        -------
        None.
        """
        # Find all the category types in the self.dataFrame and loop over them.
        for category in self.data.dtypes[self.data.dtypes == 'category'].index:
            self.data[category] = self.data[category].cat.remove_unused_categories()


    def DisplayCategoryCounts(self, categories, useMarkDown=False):
        """
        Displays all the values counts for the specified columns columns.

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.dataFrame to change the categories in.
        categories : list, array of strings
            Names of the categories to operate on.

        Returns
        -------
        None.
        """

        for category in categories:
            self.consoleHelper.PrintBoldMessage(category, useMarkDown=useMarkDown)
            display(self.data[category].value_counts())


    def DisplayAllCategoriesValueCounts(self, useMarkDown=False):
        """
        Displays the value counts of all the self.dataSeries of type "category."

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.dataFrame to operate on.

        Returns
        -------
        None.
        """
        # Find all the category types in the self.dataFrame and loop over them.
        self.DisplayCategoryCounts(self.data.dtypes[self.data.dtypes == 'category'].index, useMarkDown)


    def RemoveRowsWithLowValueCounts(self, column, criteria):
        """
        Finds the entries in "column" with low value counts and drops them.
        Performs operation in place.

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.dataFrame to operate on.
        column : string
            Column to search through.
        criteria : int
            Values counts less than this value will be dropped.

        Returns
        -------
        self.data : pandas.self.dataFrame
            self.dataFrame with the low value count rows removed.
        """
        # Get the value counts of the column.
        valueCounts = self.data[column].value_counts()

        # Extract the values that are below the threshold criteria.
        dropValues = valueCounts[valueCounts.values < criteria].index.tolist()

        # Drop the rows.
        for value in dropValues:
             self.RemoveRowByEntryValue(column, value, inPlace=True)


    def RemoveRowByEntryValue(self, column, value, inPlace=False):
        """
        Finds the locations in "column" that are equal to "value" and drops those rows.

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.dataFrame to operate on.
        column : string
            Column to search through.
        value : type of self.data in column
            Value to search for and remove.
        inPlace : bool
            Specified is the operation should be performed in place.

        Returns
        -------
        self.data : pandas.self.dataFrame
            self.dataFrame without the removed rows or None if inPlace=True.
        """

        # Gets indices of the rows we want to drop.
        dropIndices = self.data[self.data[column] == value].index.tolist()

        # Drop the rows.
        return self.data.drop(dropIndices, inplace=inPlace)


    def RemoveRowsWithValueOutsideOfCriteria(self, category, criteria, method, inPlace=False):
        """
        Replaces values beyond the criteria with the mean value for the category.  Done in place.

        Parameters
        ----------
        self.data : Pandas self.dataFrame
            The self.data.
        column: string
            Column name in the self.dataFrame.
        criteria : float
            Values below this will be removed.
        method : string
            Determines if high values or low values should be dropped.
                dropabove - Values above the criteria are removed.
                dropbelow - Values below the criteria are removed.
        inPlace : bool
            If true, the modifications are done in place.

        Returns
        -------
        self.data : pandas.self.dataFrame
            self.dataFrame without the removed rows or None if inPlace=True.
        """
        # Gets an self.dataSeries of boolean values indicating where values are outside of the range.  These are the
        # values we want to drop.
        indexMask = None
        if method == "dropabove":
            indexMask = self.data[category] > criteria
        elif method == "dropbelow":
            indexMask = self.data[category] < criteria
        else:
            raise Exception("Invalid \"method\" specified.")

        # The indexMask is a self.dataSeries that has the indexes from the original self.dataFrame and the values are the result
        # of the test statement (bools).  The indices do not necessarily correspond to the location in the self.dataFrame.  For
        # example some rows may have been removed.  Extract only the indices we want to remove by using the mask itself.
        dropIndices = indexMask[indexMask].index

        # Drop the rows.
        return self.data.drop(dropIndices, inplace=inPlace)


    def MergeCategories(self, column, fromCategories, toCategory):
        """
        Replaces every instance of a value ("from") with another in a column ("to").  Multiple from values can be
        specified at once.
        Useful for merging categories of a categorical column.
        Operation performed in place.

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.dataFrame to operate on.

        Returns
        -------
        self.data : self.dataFrame
            The new self.dataFrame with the encoded values.
        """
        for fromCategory in fromCategories:
            self.data[column] = self.data[column].replace({fromCategory : toCategory})


    def MergeNumericalDataByRange(self, column, labels, boundaries, replaceExisting=False):
        """
        Take a numerical column and groups them into categories based on range boundaries.

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.dataFrame to operate on.
        column : string
            Column name to perform the merge on.
        labels : list of strings
            A list that specifies the names for each range.
        boundaries : list of ints or floats
            A list that specifies the end points of the ranges.
        replaceExisting : bool
            If true, the existing column of self.data is replaced by the new column of merged self.data.

        Returns
        -------
        newColumnName : string
            Name of the new column that contains the categorized numbers.
        """

        newColumn      = pd.Series(np.zeros(self.data.shape[0]))
        existingColumn = self.data[column]

        for i in range(existingColumn.size):
            boundedIndices   = lendres.Algorithms.BoundingBinarySearch(existingColumn.iloc[i], boundaries, returnedUnits="indices")
            newColumn.loc[i] = labels[boundedIndices[0]]

        # Default to the original column name, then determine how to procede based on if we are to replace
        # the existing column or add a new one while retaining the original one.
        newColumnName = column
        if replaceExisting:
            self.data.drop([column], axis=1, inplace=True)
        else:
            newColumnName = column + "_categories"

        self.data[newColumnName] = newColumn.astype('category')
        return newColumnName


    def EncodeAllCategoricalColumns(self):
        """
        Converts all categorical columns (have that self.data type "category") to one hot encoded values and drops one
        value per category.  Prepares categorical columns for use in a model.

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.dataFrame to operate on.

        Returns
        -------
        None.
        """
        # Find all the category types in the self.dataFrame.
        # Gets all the columns that have the category self.data type.  That is returned as a self.dataSeries.  The
        # index (where the names are) is extracted from that.
        allCategoricalColumns = self.data.dtypes[self.data.dtypes == 'category'].index.tolist()
        self.EncodeCategoricalColumns(allCategoricalColumns)


    def EncodeCategoricalColumns(self, columns):
        """
        Converts the categorical columns "categories" to one hot encoded values and drops one value per category.
        Prepares categorical columns for use in a model.

        Parameters
        ----------
        self.data : pandas.self.dataFrame
            self.dataFrame to operate on.
        columns : list of strings
            The names of the columns to encode.

        Returns
        -------
        None.
        """
        self.data = pd.get_dummies(self.data, columns=columns, drop_first=True)