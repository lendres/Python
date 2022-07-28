"""
Created on December 27, 2021
@author: Lance A. Endres
"""
import pandas                                    as pd
import numpy                                     as np
import scipy.stats                               as stats

from   sklearn.preprocessing                     import StandardScaler
from   scipy.stats                               import zscore
from   sklearn.preprocessing                     import LabelEncoder

import os
import io

import lendres
from   lendres.ConsoleHelper                     import ConsoleHelper
from   lendres.DataHelperBase                    import DataHelperBase


class DataHelper(DataHelperBase):


    def __init__(self, fileName=None, data=None, copy=False, deep=False, consoleHelper=None):
        """
        Constructor.

        Parameters
        ----------
        fileName : stirng, optional
            Path to load the data from.  This is a shortcut for creating a DataHelper and
            then calling "LoadAndInspectData."
        data : pandas.DataFrame, optional
            DataFrame to operate on. The default is None.  If None is specified, the
            data should be loaded in a separate function call, e.g., with "LoadAndInspectData"
            or by providing a fileName to load the data from.  You cannot provide both a file
            and data.
        deep : bool, optional
            Specifies if a deep copy should be done. The default is False.  Only valid if
            the "data" parameter is specified.
        consoleHelper : ConsoleHelper
            Class the prints messages.

        Returns
        -------
        None.
        """
        super().__init__(consoleHelper)

        # Either load the data from file or the supplied existing data, but not both.
        if fileName is not None:
            self.LoadAndInspectData(fileName)

        elif data is not None:
            if copy:
                self.data = data.copy(deep)
            else:
                self.data = data


    def Copy(self, deep=False):
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
        # Creates an instance of the class by calling any subclasses's constructor.
        dataHelper                = (type(self))()
        dataHelper.consoleHelper  = self.consoleHelper
        dataHelper.labelEncoders  = self.labelEncoders.copy()

        if len(self.data) != 0:
            dataHelper.data                      = self.data.copy(deep)

        if len(self.xTrainingData) != 0:
            dataHelper.xTrainingData             = self.xTrainingData.copy(deep=deep)
            dataHelper.yTrainingData             = self.yTrainingData.copy(deep=deep)

        if len(self.xValidationData) != 0:
            dataHelper.xValidationData           = self.xValidationData.copy(deep=deep)
            dataHelper.yValidationData           = self.yValidationData.copy(deep=deep)

        if len(self.xTestingData) != 0:
            dataHelper.xTestingData              = self.xTestingData.copy(deep=deep)
            dataHelper.yTestingData              = self.yTestingData.copy(deep=deep)

        return dataHelper


    def LoadAndInspectData(self, inputFile, verboseLevel=ConsoleHelper.VERBOSEREQUESTED):
        """
        Loads a data file and performs some initial inspections and reports results.

        Parameters
        ----------
        inputFile : string
            Path and name of the file to load.

        Returns
        -------
        data : pandas.DataFrame
            Data in a pandas.DataFrame
        """

        # Validate the input file.
        if type(inputFile) != str:
            raise Exception("The input file is not a string.")

        if not os.path.exists(inputFile):
            raise Exception("The input file \"" + inputFile + "\" does not exist.")


        # Read the file in.
        self.consoleHelper.PrintTitle("Input File: " + inputFile, )
        self.data = pd.read_csv(inputFile)

        # Data size and shape.
        self.consoleHelper.PrintTitle("Data Size", verboseLevel)
        self.consoleHelper.Display(self.data.shape, verboseLevel)

        # The first few records.
        self.consoleHelper.PrintTitle("First Few Records", verboseLevel)
        self.consoleHelper.Display(self.data.head(), verboseLevel)

        # Random records.
        np.random.seed(1)
        self.consoleHelper.PrintTitle("Random Sampling", verboseLevel)
        self.consoleHelper.Display(self.data.sample(n=10), verboseLevel)

        # Data summary (mean, min, max, et cetera.
        self.consoleHelper.PrintTitle("Data Summary", verboseLevel)
        self.consoleHelper.Display(self.data.describe(), verboseLevel)

        # Check data types.
        self.PrintDataTypes(verboseLevel)

        # Check unique value counts.
        self.consoleHelper.PrintTitle("Unique Counts", verboseLevel)
        self.consoleHelper.Display(self.data.nunique(), verboseLevel)

        # See if there are any missing entries, if so they will have to be cleaned.
        self.PrintNotAvailableCounts(verboseLevel)

        return self.data


    def PrintFinalDataSummary(self):
        """
        Prints a final data summary.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.consoleHelper.PrintTitle("Data Size", ConsoleHelper.VERBOSEREQUESTED)
        self.consoleHelper.Display(self.data.shape, ConsoleHelper.VERBOSEREQUESTED)

        self.PrintDataTypes()

        self.consoleHelper.PrintTitle("Continuous Data", ConsoleHelper.VERBOSEREQUESTED)
        self.consoleHelper.Display(self.data.describe().T, ConsoleHelper.VERBOSEREQUESTED)

        if self.data.select_dtypes(include=['category']).shape[1] > 0:
            self.consoleHelper.PrintTitle("Categorical", ConsoleHelper.VERBOSEREQUESTED)
            self.consoleHelper.Display(self.data.describe(include=["category"]).T, ConsoleHelper.VERBOSEREQUESTED)


    def PrintDataTypes(self, verboseLevel=ConsoleHelper.VERBOSEREQUESTED):
        """
        Prints the data types.
        For some reasone, trying to directly print using "self.data.info()" results in the data always being displayed.  The
        printing seems to come from some other location (inside "info" itself, perhaps).  This is a work around to retrieve
        the information in a buffer and then print the data in the buffer  according to the standard operations of the ConsoleHelper.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        buffer = io.StringIO()
        self.data.info(buf=buffer)
        self.consoleHelper.PrintTitle("Data Types", verboseLevel)
        self.consoleHelper.Print(buffer.getvalue(), verboseLevel)


    def PrintNotAvailableCounts(self, verboseLevel=ConsoleHelper.VERBOSEREQUESTED):
        """
        Prints the counts of any missing (not available) entries.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        notAvailableCounts, totalNotAvailable = self.GetNotAvailableCounts()

        self.consoleHelper.PrintTitle("Missing Entry Counts", verboseLevel)
        self.consoleHelper.Display(notAvailableCounts, verboseLevel)

        if totalNotAvailable:
            self.consoleHelper.PrintWarning("Some data entries are missing.")
            self.consoleHelper.Print("Total missing: "+str(totalNotAvailable), verboseLevel)
        else:
            self.consoleHelper.Print("No entries are missing.", verboseLevel)


    def GetNotAvailableCounts(self):
        """
        Gets the counts of any missing (not available) entries.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        notAvailableCounts = self.data.isna().sum()
        totalNotAvailable  = sum(notAvailableCounts)
        return notAvailableCounts, totalNotAvailable


    def DisplayUniqueValues(self, columns):
        """
        Prints a list of the unique values in a column.

        Parameters
        ----------
        columns : list of strings
            Columns to display the counts for.

        Returns
        -------
        None.
        """
        for column in columns:
            title  = "Unique counts in \"" + column + "\":"
            self.consoleHelper.PrintTitle(title, ConsoleHelper.VERBOSEREQUESTED)

            values = pd.DataFrame(np.sort(self.data[column].unique()), columns=["Values"])
            self.consoleHelper.Display(values, ConsoleHelper.VERBOSEREQUESTED)


    def DisplayCategoryCounts(self, columns):
        """
        Displays all the values counts for the specified columns columns.

        Will not dispaly NaN values.

        Parameters
        ----------
        columns : list, array of strings
            Names of the columns to operate on.

        Returns
        -------
        None.
        """
        for column in columns:
            title  = "Category counts in \"" + column + "\":"
            self.consoleHelper.PrintTitle(title, ConsoleHelper.VERBOSEREQUESTED)

            self.consoleHelper.Display(self.data[column].value_counts(), ConsoleHelper.VERBOSEREQUESTED)


    def DisplayAllCategoriesValueCounts(self):
        """
        Displays the value counts of all columns of type "category."

        Will not dispaly NaN values.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # Find all the category types in the DataFrame and passes them to the display function.
        self.DisplayCategoryCounts(self.data.dtypes[self.data.dtypes == "category"].index)


    def ApplyTo(self, columns, function):
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
        for column in columns:
             self.data[column] = self.data[column].apply(function)


    def ChangeToCategoryType(self, columns):
        """
        Changes the specified columns to type "category."

        Parameters
        ----------
        columns : list, array of strings
            Names of the columns to change to the category data type.

        Returns
        -------
        None.
        """
        for column in columns:
            self.data[column] = self.data[column].astype("category")


    def ChangeAllObjectColumnsToCategories(self):
        """
        Changes all the columns with the "object" data type to the "category" data type.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        columnNames = self.data.dtypes[self.data.dtypes == "object"].index
        self.ChangeToCategoryType(columnNames)


    def ChangeAllCategoryColumnsToIntegers(self):
        """
        Changes all the columns with the "category" data type to the "integer" data type.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        columnNames = self.data.dtypes[self.data.dtypes == "category"].index
        self.data[columnNames] = self.data[columnNames].astype("int")


    def GetDuplicates(self, column):
        # Returns a list that has True/False values at the duplicate entries.
        duplicates         = self.dataHelper.data.duplicated(subset=[column], keep=False)

        # Gets the indices of just the "True" (duplicate) entries.
        indices            = duplicates[duplicates == True].index

        # Extract the duplicate entries from the main DataFrame and sort them by the
        # column of interest.
        duplicateDataFrame = self.dataHelper.data.iloc[indices, :].sort_values(by=[column])
        return duplicateDataFrame


    def ConvertCategoryToNumeric(self, column, trueValue):
        """
        Takes a column that has a categorical type with the categories represented
        as something other than integers and convertes it to integers.  Useful for
        converting text based columns into columns useful for model building.

        Assumes the column only has two category types so that it can be converted
        into a binary representation (0 or 1).

        This is normally done with the pandas.get_dummies function.  This function
        differs by allowing you to specify which value in the column is converted
        to true and which is converted to false.  In the pandas function, you do
        not have that choice.  It also maintains the column name.  This can be
        useful when dealing with a dependent variable.  If get_dummies is used,
        the new column name would have to be determined.

        Parameters
        ----------
        column : string
            The column to convert.
        trueValue : string or other type comparable with the "==" operator.
            The value in the column that is consided to be the "true" or "1" value.  This
            value will be output as 1, the other value as 0.

        Returns
        -------
        None.
        """
        if pd.api.types.is_categorical_dtype(self.data[column]):
            self.data[column] = self.data[column].astype("object")

        # Set the locations that have the "trueValue" as equal to one.  Have to do the not equal first.
        self.data.loc[self.data[column] != trueValue, column] = 0
        self.data.loc[self.data[column] == trueValue, column] = 1
        self.data[column] = self.data[column].astype("int")


    def DropRowsWhereDataNotAvailable(self, columns):
        """
        Drops any rows that do not have data available in the column.

        Parameters
        ----------
        columns : list of strings
            Name of the columns to look for not available entries.

        Returns
        -------
        None.
        """
        for column in columns:
            # Gets an Series of boolean values indicating where values were not available.
            indexMask = self.data[column].isna()

            # The indexMask is a Series that has the indexes from the original DataFrame and the values are the result
            # of the test statement (bools).  The indices do not necessarily correspond to the location in the DataFrame.  For
            # example some rows may have been removed.  Extract only the indices we want to remove by using the mask itself.
            dropIndices = indexMask[indexMask].index

            # Drop the rows.
            self.data.drop(dropIndices, inplace=True)


    def DropAllRowsWhereDataNotAvailable(self, inPlace=True):
        """
        Drops any rows that are missing one or more entries from the data.

        Parameters
        ----------
        inPlace : bool, optional
            If true, the modifications are done in place.  Default is true.

        Returns
        -------
        : pandas.DataFrame or None
            DataFrame without the removed rows or None if inPlace=True.
        """
        # Gets an self.dataSeries of boolean values indicating where values were not available.
        # to_numpy returns array inside of a tuple for some odd reason.  The [0] extracts the array.
        dropIndices = self.data.isna().sum(axis=1).to_numpy().nonzero()[0]

        # Drop the rows.
        return self.data.drop(dropIndices, inplace=inPlace)


    def GetRowsWithMissingEntries(self):
        """
        Gets the rows that contain missing data.

        Parameters
        ----------
        None.

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


    def ExtractLastStringTokens(self, columns):
        """
        Gets the second string token from all entries in the specified columns.

        Parameters
        ----------
        columns : list, array of strings
            Names of the columns to operate on.

        Returns
        -------
        dataFrame : pandas.DataFrame
            A DataFrame that contains the string tokens.
        """
        # Initialize variables.
        numberOfRows    = self.data.shape[0]
        dataFrame       = pd.DataFrame(index=range(numberOfRows), columns=columns)

        # Extract all the units information from the cells.  Loop over all the cells and extract the second half of the split string.
        for column in columns:
            for i in range(numberOfRows):
                value = self.data[column].iloc[i]
                dataFrame[column].iloc[i] = value.split()[1]

        return dataFrame


    def KeepFirstStringToken(self, value):
        """
        Takes a string that has multiple tokens and keeps only the first.

        Parameters
        ----------
        value : string
            The entry to be split.

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
            The entry to be split.

        Returns
        -------
        value : float
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
        Takes a value from the mileage column, strips the units string, converts the units, and returns it as a float.

        Parameters
        ----------
        value : string
            The entry to be split.

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


    def GetMinAndMaxValues(self, column, criteria, method="quantity"):
        """
        Display and maximum and minimum values in a Series.

        Parameters
        ----------
        column : string
            Names of the column to sort and display.
        criteria : list of two floats or a float
            The criteria used to determine which numbers are dropped.  See "method" for more information.
        method : string
            How to determine which values are dropped.
                percent - A percentage of the top and bottom values are dropped.
                quantity - The number of values specified is dropped.

        Returns
        -------
        : pandas.DataFrame
            A DataFrame that has both the minimum and maximum values, along with the indices where those values
            occur.  The DataFrame contains the following headings:
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
            numberOfRows = round(len(self.data[column]) * criteria / 100)
        else:
            # A boo-boo was made.
            raise Exception("Invalid \"method\" specified.")

        # Sort then display the start and end of the series.
        sortedSeries = self.data[column].sort_values()

        # Create new DataFrames for the head (smallest values) and the tail (largest values).
        # Reset the index to move the index to a column and create a new, renumbered index.  This lets us combine the two DataFrames at
        # the same index and saves the indices so we can use them later.
        # Also rename the columns to make them more meaningful.
        head = sortedSeries.head(numberOfRows).reset_index()
        head.rename({"index" : "Smallest_Index", column : "Smallest"}, axis=1, inplace=True)

        tail = sortedSeries.tail(numberOfRows).reset_index()
        tail.rename({"index" : "Largest_Index", column : "Largest"}, axis=1, inplace=True)

        # Combine the two along the columns and return the result.
        return pd.concat([head, tail], axis=1)


    def ReplaceLowOutlierWithMean(self, column, criteria):
        """
        Replaces values beyond the criteria with the mean value for the column.  Done in place.

        Parameters
        ----------
        column : string
            Column name in the data.
        criteria : float
            Criteria used to determine what is an outlier.

        Returns
        -------
        None.
        """
        mean = self.data[column].mean()
        self.data.loc[self.data[column] < criteria, column] = mean


    def DropMinAndMaxValues(self, column, criteria, method="fraction", inPlace=False):
        """
        Drops any rows that do not have data available in the column of "column."

        Parameters
        ----------
        column : string
            Names of the column to look for not available entries.
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
        data : pandas.DataFrame
            DataFrame without the removed rows or None if inPlace=True.
        """
        # This will return a struction that has the values and the indices of the minimums and maximums.
        minAndMaxValues = self.GetMinAndMaxValues(column, criteria, method=method)

        # numpy.where returns array inside of a tuple for some odd reason.  The [0] extracts the array.
        dropIndices = pd.concat([minAndMaxValues["Smallest_Index"], minAndMaxValues["Largest_Index"]])

        # Drop the rows.
        return self.data.drop(dropIndices, inplace=inPlace)


    def DropOutliers(self, columns, irqScale=1.5):
        """
        Drops any rows that are considered outliers by the definition of

        Parameters
        ----------
        columns : string or list of strings
            Names of the column(s) to have outliers dropped.
        irqScale : float
            Scale factor of interquartile range used to define outliers.

        Returns
        -------
        data : pandas.DataFrame
            DataFrame without the removed rows or None if inPlace=True.
        """
        if type(columns) != list:
            columns = [columns]

        for column in columns:
            # Get the stats we need.
            interQuartileRange = stats.iqr(self.data[column])
            limits             = np.quantile(self.data[column], q=(0.25, 0.75))

            # Set the outlier limits.
            limits[0] -= irqScale*interQuartileRange
            limits[1] += irqScale*interQuartileRange

            # Gets an Series of boolean values indicating where values are outside of the range.  These are the
            # values we want to drop.
            indexMask = (self.data[column] < limits[0]) | (self.data[column] > limits[1])

            # The indexMask is a Series that has the indexes from the original DataFrame and the values are the result
            # of the test statement (bools).  The indices do not necessarily correspond to the location in the DataFrame.  For
            # example some rows may have been removed.  Extract only the indices we want to remove by using the mask itself.
            dropIndices = indexMask[indexMask].index

            # Drop the rows.
            self.data.drop(dropIndices, inplace=True)


    def RemoveAllUnusedCategories(self):
        """
        Removes any unused categories (value count is zero) from all series that are of type "column."
        Performs operation "in place."

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # Find all the column types in the DataFrame and loop over them.
        for column in self.data.dtypes[self.data.dtypes == "category"].index:
            self.data[column] = self.data[column].cat.remove_unused_categories()


    def RemoveRowsWithLowValueCounts(self, column, criteria):
        """
        Finds the entries in "column" with low value counts and drops them.
        Performs operation in place.

        Parameters
        ----------
        column : string
            Column to search through.
        criteria : int
            Values counts less than this value will be dropped.

        Returns
        -------
        None.
        """
        # Get the value counts of the column.
        valueCounts = self.data[column].value_counts()

        # Extract the values that are below the threshold criteria.
        dropValues = valueCounts[valueCounts.values < criteria].index.tolist()

        # Drop the rows.
        for value in dropValues:
             self.RemoveRowByEntryValue(column, value, inPlace=True)


    def RemoveRowByEntryValue(self, column, value, inPlace=True):
        """
        Finds the locations in "column" that are equal to "value" and drops those rows.

        Parameters
        ----------
        column : string
            Column to search through.
        value : type of data in column
            Value to search for and remove.
        inPlace : bool
            Specified is the operation should be performed in place.

        Returns
        -------
        data : pandas.DataFrame
            DataFrame without the removed rows or None if inPlace=True.
        """
        # Gets indices of the rows we want to drop.
        dropIndices = self.data[self.data[column] == value].index.tolist()

        # Drop the rows.
        return self.data.drop(dropIndices, inplace=inPlace)


    def RemoveRowsWithValueOutsideOfCriteria(self, column, criteria, method, inPlace=False):
        """
        Replaces values beyond the criteria with the mean value for the column.  Done in place.

        Parameters
        ----------
        column: string
            Column name in the DataFrame.
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
        data : pandas.DataFrame
            DataFrame without the removed rows or None if inPlace=True.
        """
        # Gets an Series of boolean values indicating where values are outside of the range.  These are the
        # values we want to drop.
        indexMask = None
        if method == "dropabove":
            indexMask = self.data[column] > criteria
        elif method == "dropbelow":
            indexMask = self.data[column] < criteria
        else:
            raise Exception("Invalid \"method\" specified.")

        # The indexMask is a Series that has the indexes from the original DataFrame and the values are the result
        # of the test statement (bools).  The indices do not necessarily correspond to the location in the DataFrame.  For
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
        column : string
            Column name to perform the operate on.
        fromCategories : list
            List of items that will be converted.
        toCategory : string
            Item that all the "fromCategories" get converted to.

        Returns
        -------
        None.
        """
        for fromCategory in fromCategories:
            self.data[column] = self.data[column].replace({fromCategory : toCategory})


    def MergeNumericalDataByRange(self, column, labels, boundaries, replaceExisting=False):
        """
        Take a numerical column and groups them into categories based on range boundaries.

        Parameters
        ----------
        column : string
            Column name to perform the merge on.
        labels : list of strings
            A list that specifies the names for each range.
        boundaries : list of ints or floats
            A list that specifies the end points of the ranges.
        replaceExisting : bool
            If true, the existing column of data is replaced by the new column of merged data.

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

        self.data[newColumnName] = newColumn.astype("category")
        return newColumnName


    def EncodeAllCategoricalColumns(self, dropFirst=True):
        """
        Converts all categorical columns (that have data type "category") to encoded values.
        Prepares categorical columns for use in a model.

        Parameters
        ----------
        dropFirst : bool
            If true, the first category is dropped for the encoding (one hot encoding).

        Returns
        -------
        None.
        """
        # Find all the category types in the data.
        # Gets all the columns that have the category data type.  That is returned as a DataSeries.  The
        # index (where the names are) is extracted from that.
        allCategoricalColumns = self.data.dtypes[self.data.dtypes == "category"].index.tolist()
        self.EncodeCategoricalColumns(allCategoricalColumns, dropFirst)


    def EncodeCategoricalColumns(self, columns, dropFirst=True):
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
        self.data = pd.get_dummies(self.data, columns=columns, drop_first=dropFirst)


    def LabelEncodeColumns(self, columns):
        """
        Converts columns into numerically encoded values.

        Parameters
        ----------
        columns : list of strings
            The names of the columns to encode.

        Returns
        -------
        None.
        """
        if type(columns) != list:
            columns = [columns]

        for column in columns:
            labelEncoder = LabelEncoder()
            labelEncoder.fit(self.data[column])

            self.data[column]          = labelEncoder.transform(self.data[column])
            self.data[column]          = self.data[column].astype("int")
            self.labelEncoders[column] = labelEncoder


    def GetLabelsFromCodesInColumn(self, column):
        """
        Returns the text/numerical values from an encoded column.

        Parameters
        ----------
        columns : string
            The names of the column to decode.

        Returns
        -------
        None.
        """
        return self.labelEncoders[column].inverse_transform(self.data[column])


    def GetLabelsFromCodes(self, column, values):
        """
        Returns the text/numerical values from an encoded list.

        Parameters
        ----------
        columns : string
            The names of the column that was originally encoded.
        values : list, ndarray
            The values to decode.

        Returns
        -------
        None.
        """
        return self.labelEncoders[column].inverse_transform(values)


    def SplitData(self, dependentVariable, testSize, validationSize=None, stratify=False):
        """
        Splits the data.

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
        None.
        """
        # Remove the dependent varaible from the rest of the data.
        x = self.data.drop([dependentVariable], axis=1)

        # The dependent variable.
        y = self.data[dependentVariable]

        self._SplitData(x, y, testSize, validationSize, stratify)


    def ScaleData(self, columns, method="zscore"):
        """
        Scale data.

        There seems to be a bug in the "StandardScaler," it is recommened to use the "zscore" method.

        Parameters
        ----------
        method : string
            Method used to normalized the data.
            standardscaler : Uses the StandardScaler class.
            zscore : Uses the zscore.

        Returns
        -------
        None.
        """
        if len(self.xTrainingData) == 0:
            raise Exception("Data has not been split.")

        self._ScaleData(self.xTrainingData, columns, method)
        self._ScaleData(self.xTestingData, columns, method)

        if len(self.xValidationData) != 0:
            self._ScaleData(self.xValidationData, columns, method)


    def _ScaleData(self, data, columns, method):
        """
        Scale data.

        Parameters
        ----------
        method : string
            Method used to normalized the data.
            standardscaler : Uses the StandardScaler class.
            zscore : Uses the zscore.

        Returns
        -------
        None.
        """
        scaledData = data[columns].copy(deep=True)

        if method == "standardscaler":
            scaler          = StandardScaler()
            data[columns]   = pd.DataFrame(scaler.fit_transform(scaledData), columns=columns)
        elif method == "zscore":
            data[columns]   = scaledData.apply(zscore)
        else:
            raise Exception("The specified scaling method is invalid.")


    def GetSplitComparisons(self, format="countandpercentstring"):
        """
        Returns the value counts and percentages of the dependant variable for the
        original, training (if available), and testing (if available) data.

        Parameters
        ----------
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
        dependentColumn  = self.GetDependentVariableName()
        originalData     = self.data[dependentColumn]

        return self._GetSplitComparisons(originalData, format=format)


    def GetCountAndPrecentStrings(self, dataSet, dataSetName, format="countandpercentstring"):
        """
        Gets a string that is the value count of "classValue" and the percentage of the total
        that the "classValue" accounts for in the column.

        Parameters
        ----------
        classValue : int or string
            The value to count and calculate the percentage for.
        dataSet : string
            Which data set to use for the calculation.
            original - Gets the results from the original data.
            training - Gets the results from the training data.
            testing  - v the results from the test data.
        column : string, optional
            If provided, that column will be used for the calculations.  If no value is provided,
            the dependant variable is used. The default is None.

        Returns
        -------
        string : string
            The count and percentage of the total as a string.
        """
        categories         = self.data[self.GetDependentVariableName()].unique()

        valueCounts        = []
        totalCount         = len(dataSet)

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


    def GetDependentVariableName(self):
        """
        Returns the name of the dependent variable (column heading).

        Parameters
        ----------
        None.

        Returns
        -------
        : string
            The name (column heading) of the dependent variable.
        """
        if len(self.yTrainingData) == 0:
            raise Exception("The data has not been split (dependent variable not set).")

        return self.yTrainingData.name