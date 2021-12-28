# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 18:49:50 2021

@author: Lance A. Endres
"""

def ClearSpyderConsole():
    try:
        from IPython import get_ipython
        get_ipython().magic('clear')
        get_ipython().magic('reset -f')
    except:
        pass

# Automatically clear the console when this file is imported.
ClearSpyderConsole()


def PrintSectionTitle(title):
    """
    Prints a divider and title to the console.

    Parameters
    ----------
    title : string
        Title to dislay.

    Returns
    -------
    None.
    """

    quotingHashes = "######"

    # The last number is accounting for spaces.
    hashesRequired = len(title) + 2*len(quotingHashes) + 4
    hashes = ""
    for i in range(hashesRequired):
        hashes += "#"

    print("\n\n\n" + hashes)
    print(quotingHashes + "  " + title + "  " + quotingHashes)
    print(hashes)


def PrintWarning(message):
    """
    Prints warning message.

    Parameters
    ----------
    message : string
        Warning to dislay.

    Returns
    -------
    None.
    """    
    quotingHashes = "######"
    print("\n" + quotingHashes, "Warning:", message, quotingHashes)


def FormatProbabilityForOutput(probability, decimalPlaces=3):
    """
    Formats and prints a probability.  Displays it as both a fraction and a percentage.

    Parameters
    ----------
    probability : decimal
        The probability to display.
    decimalPlacess : int
        Optional, the number of digits to display (default=3).

    Returns
    -------
    None.
    """

    output = str(round(probability, decimalPlaces))
    output += " (" + str(round(probability*100, decimalPlaces-2)) + " percent)"
    return  output


def PrintTwoItemPercentages(data, category, item1Name, item2Name):
    """
    Calculates and displays precentages of each item type in a category.

    Parameters
    ----------
    data : Pandas DataFrame
        The data.
    category : string
        Name of category to use.
    item1Name : string
        Name of first type of entry in the data[category] series.
    item2Name : string
        Name of second type of entry in the data[category] series.

    Returns
    -------
    None.
    """

    counts     = data[category].value_counts()
    totalCount = data[category].count()

    item1Percent  = counts[item1Name] / totalCount
    item2Percent  = counts[item2Name] / totalCount

    print("Total entries in the \"" + category + "\" category:", totalCount)
    print("Percent of \"" + item1Name + "\":", FormatProbabilityForOutput(item1Percent))
    print("Percent of \"" + item2Name + "\":", FormatProbabilityForOutput(item2Percent))