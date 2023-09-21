"""
Created on December 29, 2021
@author: Lance A. Endres
"""
import pandas                                                   as pd
import matplotlib.pyplot                                        as plt
import seaborn                                                  as sns

from   lendres.plotting.PlotHelper                              import PlotHelper
from   lendres.plotting.AxesHelper                              import AxesHelper


class BivariateAnalysis():
    supFigureYAdjustment = 1.0


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
        axes : matplotlib.axes.Axes
            The axes of the plot.
        """

        # Must be run before creating figure or plotting data.
        PlotHelper.Format()

        # Initialize so the variable is available.
        correlationValues = []

        # If the input argument "columns" is "None," plot all the columns, otherwise, only
        # plot those columns specified in the "columns" argument.
        if columns == None:
            correlationValues = data.corr(numeric_only=True)
        else:
            correlationValues = data[columns].corr(numeric_only=True)

        axes = sns.heatmap(correlationValues, annot=True, annot_kws={"fontsize" : 10*PlotHelper.scale}, fmt=".2f")
        axes.set(title="Heat Map for Continuous Data")

        # Save it so we can return it.  Once "show" is called, the figure is no longer accessible.
        figure = plt.gcf()

        # Make sure the plot is shown.
        plt.show()

        return figure, axes


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
        PlotHelper.Format()

        if columns != None and hue != None:
            if not hue in columns:
                columns.append(hue)

        if columns == None:
            sns.pairplot(data, hue=hue)
        else:
            sns.pairplot(data[columns], hue=hue)

        # Save it so we can return it.  Once "show" is called, the figure is no longer accessible.
        figure = plt.gcf()

        figure.suptitle("Pair Plot for Continuous Data", y=1.015*BivariateAnalysis.supFigureYAdjustment)

        plt.show()

        return figure


    @classmethod
    def CreateScatterPlotComparisonByCategory(cls, data, xColumn, yColumn, sortColumn, title=None):
        """
        Creates a scatter plot of a column sorted by another column.
        Good for plotting continuous data verses continuous data and sorted by a column that is categorical.  Uses the
        hue to separate out the categories in the "sortColumn."

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
        axes : matplotlib.axes.Axes
            The axes of the plot.
        """
        if title == None:
            title = "Sorted by " + "\"" + sortColumn + "\""

        # Must be run before creating figure or plotting data.
        PlotHelper.Format()

        axes = sns.scatterplot(x=data[xColumn], y=data[yColumn], hue=data[sortColumn], palette=["indianred","mediumseagreen"])
        axes.set(title=title, xlabel=xColumn, ylabel=yColumn)
        axes.get_legend().set_title(sortColumn)

        # Save it so we can return it.  Once "show" is called, the figure is no longer accessible.
        figure = plt.gcf()

        plt.show()

        return figure, axes


    @classmethod
    def CreateComparisonPercentageBarPlot(cls, data, subCategoryColumnName, subCategoryEntries, primaryCategoryColumnName):
        """
        Creates a bar chart that shows the percentages of each type of entry of a category.

        Parameters
        ----------
        data : Pandas DataFrame
            The data.
        subCategoryColumnName : string
            Column name in the DataFrame to extract subCategoryEntries from.
        subCategoryEntries : list of strings
            Values found in the column subCategoryColumnName.
        primaryCategoryColumnName : string
            Column name in the DataFramee.  This will be the x-axis.

        Returns
        -------
        figure : Figure
            The newly created figure.
        """

        # Create data.
        data0 = cls.ExtractProportationData(data, subCategoryColumnName, subCategoryEntries[0], primaryCategoryColumnName)
        data1 = cls.ExtractProportationData(data, subCategoryColumnName, subCategoryEntries[1], primaryCategoryColumnName)

        # Combine the two for plotting.
        proportionData = pd.concat([data0, data1], ignore_index=True)

        # Must be run before creating figure or plotting data.
        PlotHelper.Format()

        # This creates the bar chart.  At the same time, save the figure so we can return it.
        #palette='winter',
        axes   = sns.barplot(x=primaryCategoryColumnName, y="Proportion", data=proportionData, hue=subCategoryColumnName)
        figure = plt.gcf()

        # Label the individual columns with a percentage, then add the titles to the plot.
        #LabelPercentagesOnCountPlot(axis, proportionData, category, scale)

        title = "\"" + primaryCategoryColumnName + "\"" + " Category"
        axes.set(title=title, xlabel=subCategoryColumnName, ylabel="Proportion")

        # Make sure the plot is shown.
        plt.show()

        return figure


    @classmethod
    def ExtractProportationData(cls, data, subCategoryColumnName, subCategoryEntry, primaryCategoryColumnName):
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
        # the entries "subCategoryEntry."
        extractedProportions = data[(data[subCategoryColumnName] == subCategoryEntry)][[subCategoryColumnName, primaryCategoryColumnName]]

        extractedProportions = extractedProportions.value_counts(normalize=True).reset_index(name="Proportion")
        extractedProportions.rename({"index" : primaryCategoryColumnName}, axis="columns", inplace=True)

        # Removes any empty categories in the column that were left over from the original data.
        extractedProportions[subCategoryColumnName] = extractedProportions[subCategoryColumnName].cat.remove_unused_categories()

        return extractedProportions

    @classmethod
    def GetCrossTabulatedValueCounts(cls, data, independentColumn, sortColumn):
        """
        Creates a DataFrame that is a matrix of the counts of categories of "independentColumn" verus "sortColumn."

        Parameters
        ----------
        data : Pandas DataFrame
            The data.
        independentColumn : string
            Column name in the DataFrame to plot on the X axis.
        sortColumn : string
            Column name of data used to calculate percentages of the categories in "independentColumn."  Typically, this is the independent
            variable of the data.  Ploted as different hues in the Y axis.

        Returns
        -------
        dataFrame : pandas.DataFrame
            A DataFrame containing a matrix of the value counts.
        """
        sorter = data[sortColumn].value_counts().index[-1]

        # The "margins" adds row/columns subtotals.
        dataFrame = pd.crosstab(data[independentColumn], data[sortColumn], margins=True).sort_values(by=sorter, ascending=False)

        return dataFrame


    @classmethod
    def CreateStackedPercentageBarPlot(cls, data, independentColumn, sortColumn, titleSuffix=None):
        """
        Creates a stacked bar chart that plots the "independentColumn" in seperate hues based on the percentages sorted by the "sortColumn."

        Parameters
        ----------
        data : Pandas DataFrame
            The data.
        independentColumn : string
            Column name in the DataFrame to plot on the X axis.
        sortColumn : string
            Column name of data used to calculate percentages of the categories in "independentColumn."  Typically, this is the independent
            variable of the data.  Ploted as different hues in the Y axis.
        titleSuffix : string or None, optional
            If supplied, the string is prepended to the title.

        Returns
        -------
        figure : Figure
            The newly created figure.
        """
        # Must be run before creating figure or plotting data.
        PlotHelper.Format()

        sorter = data[sortColumn].value_counts().index[-1]

        # The "normalize" normalizes the values to sum to 1.
        dataFrame = pd.crosstab(data[independentColumn], data[sortColumn], normalize="index").sort_values(by=sorter, ascending=False)

        axes = dataFrame.plot(kind="bar", stacked=True)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        title = "\"" + independentColumn + "\"" + " as Fraction of " + "\"" + sortColumn + "\""
        AxesHelper.Label(axes, title=title, xLabel=independentColumn, yLabels="Fraction of "+sortColumn, titleSuffix=titleSuffix)

        figure = plt.gcf()
        plt.show()

        return figure


    @classmethod
    def CreateDistributionByTargetPlot(cls, data, independentColumn, sortColumn, titleSuffix=None):
        """
        Creates histograms of the "independentColumn" sorted by the "sortColumn."

        Parameters
        ----------
        data : Pandas DataFrame
            The data.
        independentColumn : string
            Column name in the DataFrame to plot on the X axis.
        sortColumn : string
            Column name of data used to calculate percentages of the categories in "independentColumn."  Typically, this is the independent
            variable of the data.  Ploted as different hues in the Y axis.
        titleSuffix : string or None, optional
            If supplied, the string is prepended to the title.

        Returns
        -------
        figure : Figure
            The newly created figure.
        """
        # Unique values in sorting column.  Sort them so they are presented correctly (in order) and
        # so the colors match those used in CreateBoxPlotByTarget.
        uniqueSortValues = data[sortColumn].unique()
        uniqueSortValues.sort()


        # Number of unique values.
        numberOfUniqueValues = uniqueSortValues.size

        # Must be run before creating figure or plotting data.
        PlotHelper.Format(width=6*numberOfUniqueValues, height=6)

        # Create figure and a row of axes.
        figure, axeses = plt.subplots(1, numberOfUniqueValues)

        for i in range(numberOfUniqueValues):
            sns.histplot(
                data=data[data[sortColumn] == uniqueSortValues[i]],
                x=independentColumn,
                kde=True,
                ax=axeses[i],
                color=sns.color_palette()[i]
            )
            title = "Distribution for " + "\""+ sortColumn + "\"" + " = " + str(uniqueSortValues[i])
            axeses[i].set(title=title, xlabel=independentColumn)

        figure.suptitle("\"" + independentColumn + "\"" + " Separated by " + "\"" + sortColumn + "\"", y=0.92*BivariateAnalysis.supFigureYAdjustment)

        plt.tight_layout()
        plt.show()

        return figure


    @classmethod
    def CreateBoxPlotByTarget(cls, data, independentColumn, sortColumn, titleSuffix=None):
        """
        Creates histograms and bar plots of the "independentColumn" sorted by the "sortColumn."

        Parameters
        ----------
        data : Pandas DataFrame
            The data.
        independentColumn : string
            Column name in the DataFrame to plot on the x-axis.
        sortColumn : string
            Column name of data used to calculate percentages of the categories in "independentColumn."  Typically, this is the independent
            variable of the data.  Ploted as different hues in the y-axis.
        titleSuffix : string or None, optional
            If supplied, the string is prepended to the title.

        Returns
        -------
        figure : Figure
            The newly created figure.
        """
        # Must be run before creating figure or plotting data.
        PlotHelper.Format(width=12, height=6)

        # Create figure and a 2x2 grid of axes.
        figure, axeses = plt.subplots(1, 2)

        # Box plot with outliers.
        sns.boxplot(data=data, x=sortColumn, y=independentColumn, ax=axeses[0])
        title = "Boxplot with Outliers"
        axeses[0].set(title=title, xlabel=sortColumn, ylabel=independentColumn)

        # Box plot without outliers.
        sns.boxplot(data=data, x=sortColumn, y=independentColumn, ax=axeses[1], showfliers=False)
        title = "Boxplot without Outliers"
        axeses[1].set(title=title, xlabel=sortColumn, ylabel=independentColumn)

        figure.suptitle("\"" + independentColumn + "\"" + " Separated by " + "\"" + sortColumn + "\"", y=0.92*BivariateAnalysis.supFigureYAdjustment)

        plt.tight_layout()
        plt.show()

        return figure