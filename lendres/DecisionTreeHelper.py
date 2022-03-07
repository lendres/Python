# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 07:49:25 2022

@author: Lance
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

import os

import lendres
from lendres.CategoricalRegressionHelper import CategoricalRegressionHelper

class DecisionTreeHelper(CategoricalRegressionHelper):

    def __init__(self, dataHelper):
        """
        Constructor.

        Parameters
        ----------
        dataHelper : DataHelper
            DataHelper that has the data in a pandas.DataFrame.

        Returns
        -------
        None.
        """
        super().__init__(dataHelper)


    def CreateModel(self, **kwargs):
        """
        Creates a decision tree model.

        Parameters
        ----------
        **kwargs : keyword arguments
            These arguments are passed on to the DecisionTreeClassifier.

        Returns
        -------
        None.
        """

        if len(self.xTrainingData) == 0:
            raise Exception("The data has not been split.")

        self.model = DecisionTreeClassifier(**kwargs, random_state=1)
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


    def SaveTreeAsText(self, fileNameForExport):
        """
        Saves the decision tree as a text file.

        Parameters
        ----------
        fileNameForExport : string
            The file name for exporting.  If a complete path is provided, it is used.
            Otherwise, the current directory is used.

        Returns
        -------
        None.
        """
        # Make sure the file name was passed as a string.
        if not isinstance(fileNameForExport, str):
            raise Exception(("File must be provided and must be a string."))

        # The data for exporting.
        treeText = self.GetTreeAsText()

        # Extract the file extension from the path, if it exists.
        fileName, fileExtension = os.path.splitext(fileNameForExport)

        # Ensure the file extension is for a text file.
        if fileExtension != ".txt":
            fileNameForExport += ".txt"

        # Write the file.
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