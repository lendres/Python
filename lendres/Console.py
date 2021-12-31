# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 18:49:50 2021

@author: Lance A. Endres
"""

import IPython


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


def PrintWarning(message, useMarkDown=False):
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
    PrintBoldMessage("Warning: " + message, useMarkDown)


def PrintBoldMessage(message, useMarkDown=False):
    """
    Prints a message.  If markdown is enable it will use markdown to make the message bold.  If markdown is not used,
    it will use asterisks to help the text stand out.

    Parameters
    ----------
    message : string
        Warning to dislay.
    useMarkDown : bool
        If true, markdown output is enabled.

    Returns
    -------
    None.
    """
    quotingNotation = "***"

    if useMarkDown:
        # Don't use spaces between the asterisks and message so it prints bold in markdown.
        IPython.display.display(IPython.display.Markdown(quotingNotation + message + quotingNotation))
    else:
        # Use the ","s in the print function to add spaces between the asterisks and message.  For plain text
        # output, this makes it more readable.
        print(quotingNotation, message, quotingNotation)


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