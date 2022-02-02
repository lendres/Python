# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 18:06:31 2021

@author: Lance
"""
import pandas as pd
import numpy as np
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
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to change the categories in.
    category : string
        Names of the category to look for not available entries.

    Returns
    -------
    DataFrame without the removed rows or None if inPlace=True.
    """
    # Gets an DataSeries of boolean values indicating where values were not available.
    notAvailableMask = data["Price"].isna()

    # numpy.where returns array inside of a tuple for some odd reason.  The [0] extracts the array.
    dropIndices = np.where(notAvailableMask)[0]

    # Drop the rows.
    if inPlace:
        data.drop(dropIndices, inplace=inPlace)
    else:
        return data.drop(dropIndices)