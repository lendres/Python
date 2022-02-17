# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 07:49:25 2022

@author: Lance
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

import os

import lendres
from lendres.CategoricalRegressionHelper import CategoricalRegressionHelper

class DecisionTreeHelper(CategoricalRegressionHelper):

    def __init__(self, data):
        """
        Constructor.
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame to operate on.
            
        Returns
        -------
        None.
        """
        super().__init__(data)
        self.gridSearch = None


    def CreateModel(self, max_depth=None, **kwargs):
        """
        Creates a linear regression model.  Splits the data and creates the model.

        Parameters
        ----------
        data : pandas.DataFrame
            Data in a pandas.DataFrame
        dependentVariable : string
            Name of the column that has the dependant data.
        testSize : double
            Fraction of the data to use as test data.  Must be in the range of 0-1.

        Returns
        -------
        data : pandas.DataFrame
            Data in a pandas.DataFrame
        """

        if len(self.xTrainingData) == 0:
            raise Exception("The data has not been split.")

        self.model = DecisionTreeClassifier(max_depth=max_depth, **kwargs, random_state=1)
        self.model.fit(self.xTrainingData, self.yTrainingData)


    def CreateGridSearchModel(self, parameters, scoringFunction, cv=5, **kwargs):
        if len(self.xTrainingData) == 0:
            raise Exception("The data has not been split.")

        self.model = DecisionTreeClassifier(**kwargs, random_state=1)


        # Type of scoring used to compare parameter combinations.
        scorer = metrics.make_scorer(scoringFunction)

        # Run the grid search.
        self.gridSearch = GridSearchCV(self.model, parameters, scoring=scorer, cv=cv)
        self.gridSearch = self.gridSearch.fit(self.xTrainingData, self.yTrainingData)

        # Set the model (classifier) to the best combination of parameters.
        self.model = self.gridSearch.best_estimator_

        # Fit the best algorithm to the data.
        self.model.fit(self.xTrainingData, self.yTrainingData)


    def CreateDecisionTreePlot(self, scale=1.0):
        """
        Plots the decision tree.

        Parameters
        ----------
        scale : double
            Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot.

        Returns
        -------
        None.
        """
        plt.figure(figsize=(20,30))
        featureNames = list(self.xTrainingData.columns)
        tree.plot_tree(self.model, feature_names=featureNames, filled=True, fontsize=9*scale, node_ids=True, class_names=True)
        plt.show()


    def CreateFeatureImportancePlot(self, scale=1.0):
        """
        Plots importance factors as a bar plot.

        Parameters
        ----------
        scale : double
            Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot.

        Returns
        -------
        None.
        """
        # Need the values in the reverse order (smallest to largest) for the bar plot to get the largest value on
        # the top (highest index position).
        importancesDataFrame = self.GetSortedImportance(ascending=True)
        indices              = range(importancesDataFrame.shape[0])

        lendres.Plotting.FormatPlot(scale=scale)

        plt.barh(indices, importancesDataFrame["Importance"], color="cornflowerblue", align="center")
        plt.yticks(indices, importancesDataFrame.index, fontsize=12*scale)
        plt.gca().set(title="Feature Importances", xlabel="Relative Importance")

        plt.show()


    def GetSortedImportance(self, ascending=False):
        """
        Sorts the importance factors and returns them in a Pandas DataFrame.

        Parameters
        ----------
        ascending : bool
            Specifies if the values should be sorted as ascending or descending.

        Returns
        -------
        : pandas.DataFrame
            DataFrame of the sorted importance values.
        """
        return pd.DataFrame(self.model.feature_importances_,
                            columns=["Importance"],
                            index=self.xTrainingData.columns).sort_values(by="Importance", ascending=ascending)


    def GetTreeAsText(self):
        """
        Gets the decision tree as a string.

        Parameters
        ----------
        None.

        Returns
        -------
        treeText : string
            The decision tree as a string.
        """
        featureNames = list(self.xTrainingData.columns)
        treeText     = tree.export_text(self.model, feature_names=featureNames, show_weights=True)
        return treeText


    def SaveTreeAsText(self, fileNameForExport=None):

        if not isinstance(fileNameForExport, str):
            raise Exception(("File must be provided and must be a string."))

        treeText = self.GetTreeAsText()

        fileName, fileExtension = os.path.splitext(fileNameForExport)

        if fileExtension != ".txt":
            fileNameForExport += ".txt"

        file = open(fileNameForExport, "w")
        file.write(treeText)
        file.close()



# # Decision Tree arrows are missing, how to fix this? Use the following code as your reference to resolve the issue
# plt.figure(figsize=(20,30))
# out = tree.plot_tree(model,feature_names=feature_names,filled=True,fontsize=9,node_ids=False,class_names=None,)
# #below code will add arrows to the decision tree split if they are missing
# for o in out:
#      arrow = o.arrow_patch
#      if arrow is not None:
#         arrow.set_edgecolor('black')
#         arrow.set_linewidth(1)
# plt.show()