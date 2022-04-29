# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas                as pd
import matplotlib.pyplot     as plt
import seaborn               as sns

from lendres.PlotHelper      import PlotHelper

class BivariateAnalysis():

    @classmethod
    def CreateBivariateHeatMap(cls, data, columns=None):
        """
        Creates a new figure that has a bar plot labeled with a percentage for a single variable analysis.  Does this
        for every entry in the list of columns.

        Parameters
        ----------
        data : pandas.DataFrame
            The data.
        columns : list of strings
            If specified, only those columns are used for the correlation, otherwise all numeric columns will be used.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        axis : matplotlib.pyplot.axis
            The axis of the plot.
        """

        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        # Initialize so the variable is available.
        correlationValues = []

        # If the input argument "columns" is "None," plot all the columns, otherwise, only
        # plot those columns specified in the "columns" argument.
        if columns == None:
            correlationValues = data.corr()
        else:
            correlationValues = data[columns].corr()

        axis = sns.heatmap(correlationValues, annot=True, annot_kws={"fontsize" : 10*PlotHelper.scale}, fmt=".2f")
        axis.set(title="Heat Map for Continuous Data")

        # Save it so we can return it.  Once "show" is called, the figure is no longer accessible.
        figure = plt.gcf()

        # Make sure the plot is shown.
        plt.show()

        return figure, axis


    @classmethod
    def CreateBivariatePairPlot(cls, data, columns=None, hue=None):
        """
        Creates a new figure that has a bar plot labeled with a percentage for a single variable analysis.  Does this
        for every entry in the list of columns.

        Parameters
        ----------
        data : Pandas DataFrame
            The data.
        columns : List of strings
            If specified, only those columns are used for the correlation, otherwise all numeric columns will be used.
        save : bool, optional
            If true, the plots as images.  The default is False.
        **kwargs : keyword arguments
            These arguments are passed on to the DecisionTreeClassifier.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        """

        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        if columns != None and hue != None:
            if not hue in columns:
                columns.append(hue)

        if columns == None:
            sns.pairplot(data, hue=hue)
        else:
            sns.pairplot(data[columns], hue=hue)

        # Save it so we can return it.  Once "show" is called, the figure is no longer accessible.
        figure = plt.gcf()

        figure.suptitle("Pair Plot for Continuous Data", y=1.01)

        plt.show()

        return figure


    @classmethod
    def PlotComparisonByCategory(cls, data, xColumn, yColumn, sortColumn, title):
        """
        Creates a scatter plot of a column sorted by another column.

        Parameters
        ----------
        data : Pandas DataFrame
            The data.
        xColumn : string
            Independent variable column in the data.
        yColumn : string
            Dependent variable column in the data.
        sortColumn : string
            Variable column in the data to sort by.
        title : string
            Plot title.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        axis : matplotlib.pyplot.axis
            The axis of the plot.
        """
        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        axis = sns.scatterplot(x=data[xColumn], y=data[yColumn], hue=data[sortColumn], palette=["indianred","mediumseagreen"])
        axis.set(title=title, xlabel=xColumn.title(), ylabel=yColumn.title())
        axis.get_legend().set_title(sortColumn.title())

        # Save it so we can return it.  Once "show" is called, the figure is no longer accessible.
        figure = plt.gcf()

        plt.show()

        return figure, axis


    @classmethod
    def CreateComparisonPercentageBarPlot(cls, data, primaryCategoryName, primaryCategoryEntries, subCategoryName):
        """
        Creates a bar chart that shows the percentages of each type of entry of a category.

        Parameters
        ----------
        data : Pandas DataFrame
            The data.
        category: string
            Category name in the DataFrame.

        Returns
        -------
        figure : Figure
            The newly created figure.
        """

        # Create data.
        data0 = cls.ExtractProportationData(data, primaryCategoryName, primaryCategoryEntries[0], subCategoryName)
        data1 = cls.ExtractProportationData(data, primaryCategoryName, primaryCategoryEntries[1], subCategoryName)

        # Combine the two for plotting.
        proportionData = pd.concat([data0, data1], ignore_index=True)

        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

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


    @classmethod
    def ExtractProportationData(cls, data, primaryCategoryName, primaryCategoryEntry, subCategoryName):
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