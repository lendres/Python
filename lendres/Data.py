# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 18:06:31 2021

@author: Lance
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
from IPython.display import display

import lendres

def LoadAndInspectData(inputFile):
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

    print("Input file:", inputFile)

    data = pd.read_csv(inputFile)

    print("\nData size:", data.shape)

    print("\nFirst few records:")
    display(data.head())

    print("\nRecord random sampling:")
    np.random.seed(1)
    display(data.sample(n=10))

    print("\nData summary:")
    display(data.describe())

    # Check data types.
    print("\nData types:")
    display(data.info())

    # Check unique value counts.
    print("\nUnique counts:")
    print(data.nunique())

    # See if there are any missing entries, if so they will have to be cleaned.
    PrintNotAvailableCounts(data)

    return data


def PrintNotAvailableCounts(data):
    """
    Prints the counts of any missing (not available) entries.

    Parameters
    ----------
    data : pandas.DataFrame
        Data in a pandas.DataFrame

    Returns
    -------
    None.
    """
    print("\nMissing entry counts:")
    notAvailableCounts = data.isna().sum()
    print(notAvailableCounts.to_string())

    totalNotAvailable = sum(notAvailableCounts)
    if totalNotAvailable:
        lendres.Console.PrintWarning("Some data entries are missing.")
        print("Total missing:", totalNotAvailable)
    else:
        print("No entries are missing.")


def ChangeToCategory(data, categories):
    """
    Changes the data series specified to type "category."
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to change the categories in.
    categories : list, array of strings
        Names of the categories to change to the category data type.

    Returns
    -------
    None.
    """

    for category in categories:
        data[category] = data[category].astype("category")


def DropRowsWhereDataNotAvailable(data, category, inPlace=False):
    """
    Drops any rows that do not have data available in the category of "category."
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to change the categories in.
    category : string
        Names of the category to look for not available entries.
    inPlace : bool
        If true, the modifications are done in place.

    Returns
    -------
    DataFrame without the removed rows or None if inPlace=True.
    """
    # Gets an DataSeries of boolean values indicating where values were not available.
    indexMask = data[category].isna()
    
    # The indexMask is a DataSeries that has the indexes from the original DataFrame and the values are the result
    # of the test statement (bools).  The indices do not necessarily correspond to the location in the DataFrame.  For
    # example some rows may have been removed.  Extract only the indices we want to remove by using the mask itself.
    dropIndices = indexMask[indexMask].index
    
    # Drop the rows.
    return data.drop(dropIndices, inplace=inPlace)
    

def DropAllRowsWhereDataNotAvailable(data, inPlace=False):
    """
    Drops any rows that are missing one or more entries from the data.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to change the categories in.
    inPlace : bool
        If true, the modifications are done in place.

    Returns
    -------
    DataFrame without the removed rows or None if inPlace=True.
    """
    # Gets an DataSeries of boolean values indicating where values were not available.
    # to_numpy returns array inside of a tuple for some odd reason.  The [0] extracts the array.
    dropIndices = data.isna().sum(axis=1).to_numpy().nonzero()[0]

    # Drop the rows.
    return data.drop(dropIndices, inplace=inPlace)

    
def GetRowsWithMissingEntries(data):
    """
    Gets the rows that contain missing data.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to change the categories in.
        
    Returns
    -------
    A list that contains the indices of the rows that contain at least one missing entry.
    """
    
    locations = data.isna().sum(axis=1).to_numpy().nonzero()[0]
    count     = len(locations)
        
    return locations, count


def ExtractLastStringTokens(data, categories):
    """
    Gets last string token from all the categories.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to change the categories in.
    categories : list, array of strings
        Names of the categories to operate on.

    Returns
    -------
    A list that contains the indices of the rows that contain at least one missing entry.
    """
    # Initialize variables.
    numberOfRows  = data.shape[0]
    dataFrame     = pd.DataFrame(index=range(numberOfRows), columns=categories)

    # Extract all the units information from the cells.  Loop over all the cells and extract the second half of the split string.
    for category in categories:
        for i in range(numberOfRows):
            value = data[category].iloc[i]
            dataFrame[category].iloc[i] = value.split()[1]
    
    return dataFrame


def KeepFirstStringToken(value):
    """
    Takes a string that has multiple tokens and keeps only the first.

    Parameters
    ----------
    value : string
        The data entry.

    Returns
    -------
    The first token of the string.
    """
    
    # Make sure we are processing a string.
    if isinstance(value, str):
        # Splits the string at the space and returns the first entry as a number.
        return value.split()[0]
    
    else:
        # Entry wasn't a string, return an empty string.
        return ""


def KeepFirstTokenAsNumber(value):
    """
    Takes a string that has two tokens (a number followed by a string), extracts the numerical part, and returns it as a float.

    Parameters
    ----------
    value : string
        The data entry.

    Returns
    -------
    float
        The first token of the string as a number.
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


def ConvertMileage(value):
    """
    Takes a value from the mileage category, strips the units string, converts the units, and returns it as a float.

    Parameters
    ----------
    value : string
        The data entry.

    Returns
    -------
    float
        The mileage as km per liter.
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


def GetMinAndMaxValues(data, category, criteria, method="quantity"):
    """
    Display and maximum and minimum values in a DataSeries.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to change the categories in.
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
    pandas.DataFrame
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
        numberOfRows = round(len(data[category]) * criteria / 100)
    else:
        # A boo-boo was made.
        raise Exception("Invalid \"method\" specified.")

    # Sort then display the start and end of the series.
    sortedSeries = data[category].sort_values()
    
    # Create new DataFrames for the head (smallest values) and the tail (largest values).
    # Reset the index to move the index to a column and create a new, renumbered index.  This lets us combine the two DataFrames at
    # the same index and saves the indices so we can use them later.
    # Also rename the columns to make them more meaningful.
    head = sortedSeries.head(numberOfRows).reset_index()
    head.rename({"index" : "Smallest_Index", category : "Smallest"}, axis=1, inplace=True)
    
    tail = sortedSeries.tail(numberOfRows).reset_index()
    tail.rename({"index" : "Largest_Index", category : "Largest"}, axis=1, inplace=True)

    # Combine the two along the columns and return the result.
    return pd.concat([head, tail], axis=1)


def ReplaceLowOutlierWithMean(data, category, criteria):
    """
    Replaces values beyond the criteria with the mean value for the category.  Done in place.

    Parameters
    ----------
    data : Pandas DataFrame
        The data.
    category: string
        Category name in the DataFrame.

    Returns
    -------
    None.
    """
    mean = data[category].mean()
    data.loc[data[category] < criteria, category] = mean
    

def DropMinAndMaxValues(data, category, criteria, method="fraction", inPlace=False):
    """
    Drops any rows that do not have data available in the category of "category."
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to change the categories in.
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
    DataFrame without the removed rows or None if inPlace=True.
    """

    # This will return a struction that has the values and the indices of the minimums and maximums.
    minAndMaxValues = GetMinAndMaxValues(data, category, criteria, method=method)
        
    # numpy.where returns array inside of a tuple for some odd reason.  The [0] extracts the array.
    dropIndices = pd.concat([minAndMaxValues["Smallest_Index"], minAndMaxValues["Largest_Index"]])
    
    # Drop the rows.
    return data.drop(dropIndices, inplace=inPlace)

    
def DropOutliers(data, category, irqScale=1.5, inPlace=False):
    """
    Drops any rows that are considered outliers by the definition of
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to change the categories in.
    category : string
        Names of the category to look for not available entries.
    irqScale : float
        Scale factor of interquartile range used to define outliers.
    inPlace : bool
        If true, the modifications are done in place.

    Returns
    -------
    DataFrame without the removed rows or None if inPlace=True.
    """

    # Get the stats we need.
    interQuartileRange = stats.iqr(data[category])
    limits             = np.quantile(data[category], q=(0.25, 0.75))
    
    # Set the outlier limits.
    limits[0] -= irqScale*interQuartileRange
    limits[1] += irqScale*interQuartileRange
    
    # Gets an DataSeries of boolean values indicating where values are outside of the range.  These are the
    # values we want to drop.
    indexMask = (data[category] < limits[0]) | (data[category] > limits[1])

    # The indexMask is a DataSeries that has the indexes from the original DataFrame and the values are the result
    # of the test statement (bools).  The indices do not necessarily correspond to the location in the DataFrame.  For
    # example some rows may have been removed.  Extract only the indices we want to remove by using the mask itself.
    dropIndices = indexMask[indexMask].index
    
    # Drop the rows.
    return data.drop(dropIndices, inplace=inPlace)


def RemoveAllUnusedCategories(data):
    """
    Removes any unused categories (value count is zero) from all series that are of type "category."
    Performs operation "in place."

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to operate on.

    Returns
    -------
    None.
    """
    # Find all the category types in the DataFrame and loop over them.
    for category in data.dtypes[data.dtypes == 'category'].index:
        data[category] = data[category].cat.remove_unused_categories()


def DisplayCategoryCounts(data, categories, useMarkDown=False):
    """
    Displays all the values counts for the specified columns columns.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to change the categories in.
    categories : list, array of strings
        Names of the categories to operate on.

    Returns
    -------
    None.
    """

    for category in categories:
        lendres.Console.PrintBoldMessage(category, useMarkDown=useMarkDown)
        display(data[category].value_counts())
   
    
def DisplayAllCategoriesValueCounts(data, useMarkDown=False):
    """
    Displays the value counts of all the DataSeries of type "category."

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to operate on.

    Returns
    -------
    None.
    """
    # Find all the category types in the DataFrame and loop over them.
    DisplayCategoryCounts(data, data.dtypes[data.dtypes == 'category'].index, useMarkDown)


def RemoveRowsWithLowValueCounts(data, column, criteria):
    """
    Finds the entries in "column" with low value counts and drops them.
    Performs operation in place.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to operate on.
    column : string
        Column to search through.
    criteria : int
        Values counts less than this value will be dropped.

    Returns
    -------
    DataFrame with the low value count rows removed.
    """
    # Get the value counts of the column.
    valueCounts = data[column].value_counts()
    
    # Extract the values that are below the threshold criteria.
    dropValues = valueCounts[valueCounts.values < criteria].index.tolist()
    
    # Drop the rows.
    for value in dropValues:
         RemoveRowByEntryValue(data, column, value, inPlace=True)


def RemoveRowByEntryValue(data, column, value, inPlace=False):
    """
    Finds the locations in "column" that are equal to "value" and drops those rows.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to operate on.
    column : string
        Column to search through.
    value : type of data in column
        Value to search for and remove.
    inPlace : bool
        Specified is the operation should be performed in place.

    Returns
    -------
    DataFrame without the removed rows or None if inPlace=True.
    """
        
    # Gets indices of the rows we want to drop.
    dropIndices = data[data[column] == value].index.tolist()
    
    # Drop the rows.
    return data.drop(dropIndices, inplace=inPlace)


def MergeCategories(data, column, fromCategories, toCategory):
    """
    Replaces every instance of a value ("from") with another in a column ("to").  Multiple from values can be
    specified at once.
    Useful for merging categories of a categorical column.
    Operation performed in place.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to operate on.

    Returns
    -------
    data : DataFrame
        The new DataFrame with the encoded values.
    """
    for fromCategory in fromCategories:
        data[column] = data[column].replace({fromCategory : toCategory})