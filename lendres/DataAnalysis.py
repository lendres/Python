# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 21:50:49 2022

@author: Lance
"""

import pandas as pd
import matplotlib as plt
import seaborn as sns

import lendres

def CreateComparisonPercentageBarPlot(data, primaryCategoryName, primaryCategoryEntries, subCategoryName, scale=1.0):
    """
    Creates a bar chart that shows the percentages of each type of entry of a category.

    Parameters
    ----------
    data : Pandas DataFrame
        The data.
    category: string
        Category name in the DataFrame.
    scale : double
        Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot.

    Returns
    -------
    figure : Figure
        The newly created figure.
    """

    # Create data.
    data0 = ExtractProportationData(data, primaryCategoryName, primaryCategoryEntries[0], subCategoryName)
    data1 = ExtractProportationData(data, primaryCategoryName, primaryCategoryEntries[1], subCategoryName)
    
    # Combine the two for plotting.
    proportionData = pd.concat([data0, data1], ignore_index=True)
    
    # Must be run before creating figure or plotting data.
    lendres.Plotting.FormatPlot(scale=scale)

    # This creates the bar chart.  At the same time, save the figure so we can return it.
    #palette='winter',
    axis   = sns.barplot(x=subCategoryName, y="Proportion", data=proportionData, hue=primaryCategoryName)
    figure = plt.gcf()

    # Label the individual columns with a percentage, then add the titles to the plot.
    #LabelPercentagesOnCountPlot(axis, proportionData, category, scale)

    title = "\"" + subCategoryName.title() + "\"" + " Category"
    axis.set(title=title, xlabel=primaryCategoryName.title(), ylabel="Proportion")

    # Make sure the plot is shown.
    plt.show()

    return figure

def ExtractProportationData(data, primaryCategoryName, primaryCategoryEntry, subCategoryName):
    """
    Extracts and calculates proportional data.

    Parameters
    ----------
    data : Pandas DataFrame
        The data.
 

    Returns
    -------
    extractedProportions : Pandas DataFrame
        A DataFrame similar to the one below where a primary category is extracted and then broken down
        by a sub-category.  The proportion of each sub-category is calculated and in a column called "Proportion."
        
        In the example below, the "Product" would be the primary category and "Gender" would be the sub-category.
    
      Product  Gender  Proportion
    0   TM798    Male       0.825
    1   TM798  Female       0.175
    """
    # Retrieve the primary category and sub-category columns for the rows where the primary category contain
    # the entries "primaryCategoryEntry."
    extractedProportions = data[(data[primaryCategoryName] == primaryCategoryEntry)][[primaryCategoryName, subCategoryName]]
    
    extractedProportions = extractedProportions.value_counts(normalize=True).reset_index(name="Proportion")
    extractedProportions.rename({"index" : subCategoryName}, axis="columns", inplace=True)
    
    # Removes any empty categories in the column that were left over from the original data.
    extractedProportions[primaryCategoryName] = extractedProportions[primaryCategoryName].cat.remove_unused_categories()
    
    return extractedProportions


def MakeComparisonPercentageBarPlots(data, primaryCategoryName, primaryCategoryEntries, subCategories, save=False):
    """
    Creates a new figure that has a bar plot labeled with a percentage for a single variable analysis.  Does this
    for every entry in the list of categories.

    Parameters
    ----------
    data : Pandas DataFrame
        The data.
    category : an arry or list of strings
        Category names in the DataFrame.
    save : bool
        If true, the plots are saved to the default plotting directory.

    Returns
    -------
    None.
    """
    for subCategory in subCategories:
        figure = CreateComparisonPercentageBarPlot(data, primaryCategoryName, primaryCategoryEntries, subCategory)

        if save:
            fileName  = "Percentage Bar Plot " + subCategory.title() + " Category"
            lendres.Plotting.SavePlot(fileName, figure=figure, useDefaultOutputFolder=True)