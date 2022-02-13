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


questionNumber = 0

def PrintQuestionTitle(number=None):
    """
    Prints a divider with the question number to the console.

    Parameters
    ----------
    None.

    Returns
    -------
    None.
    """

    global questionNumber

    if number == None:
        questionNumber += 1
    else:
        questionNumber = number

    PrintSectionTitle("Question " + str(questionNumber))


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


def PrintHypothesisTestResult(nullHypothesis, alternativeHypothesis, pValue, levelOfSignificance=0.05, precision=4, useMarkDown=False):
    """
    Prints the result of a hypothesis test.

    Parameters
    ----------
    nullHypothesis
    data : Pandas DataFrame
        The data.
    alternativeHypothesis : string
        A string that specifies what the alternative hypothesis is.
     pValue : double
        The p-value output from the statistical test.
    levelOfSignificance : double
        Name of second type of entry in the data[category] series.
    precision : int
        The number of significant digits to display for the p-value.
    useMarkDown : bool
        If true, markdown output is enabled.

    Returns
    -------
    None.
    """

    # Display the input values that will be compared.  This ensures the values can be checked so that
    # no mistake was made when entering the information.  The raw p-value is output so it can be examined without
    # any formatting that may obscure the value.  The the values are output in an easier to read format.
    print("Raw p-value:", pValue)
    print("\nP-value:            ", round(pValue, precision))
    print("Level of significance:", round(levelOfSignificance, precision))

    # Check the test results and print the message.
    if pValue < levelOfSignificance:
        PrintBoldMessage("The null hypothesis CAN be rejected.", useMarkDown)
        print(alternativeHypothesis)
    else:
        PrintBoldMessage("The null hypothesis CAN NOT be rejected.", useMarkDown)
        print(nullHypothesis)