# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 18:06:31 2021

@author: Lance
"""
import pandas as pd
import lendres.Console

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
    print(data.head())
    
    print("\nData description:")
    print(data.describe())
    
    # Check data types.
    print("\nData types:")
    print(data.info())
    
    # See if there are any missing entries, if so they will have to be cleaned.
    print("\nLook for any entries that are missing:")
    notAvailableCounts = data.isna().sum()
    print(notAvailableCounts)
    
    if sum(notAvailableCounts) != 0:
        lendres.Console.PrintWarning("Some data entries are missing.")
    
    return data