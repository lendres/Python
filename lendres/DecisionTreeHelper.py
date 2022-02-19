# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 07:49:25 2022

@author: Lance
"""

import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display

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
        self.gridSearch              = None
        self.CostComplexityPath      = None
        self.decisionTreeHelpers = None
        

    @classmethod    
    def FromData(cls, original, deep=False):
        decisionTreeHelper = DecisionTreeHelper(None)
        decisionTreeHelper.CopyData(original, deep)
        return decisionTreeHelper


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


    def CreateGridSearchModel(self, parameters, scoringFunction, displayBestParameters=True, **kwargs):
        """
        Creates a cross validation grid search model.

        Parameters
        ----------
        parameters : dictionary
            Grid search parameters.
        scoringFunction : function
            Method use to calculate a score for the model.
        displayBestParameters : bool
            If true, it outputs the parameters of the selected model.
        **kwargs : keyword arguments
            These arguments are passed on to the GridSearchCV.

        Returns
        -------
        None.

        """
        # Make sure there is data to operate on.
        if len(self.xTrainingData) == 0:
            raise Exception("The data has not been split.")

        # Type of scoring used to compare parameter combinations.
        scorer = metrics.make_scorer(scoringFunction)

        # Run the grid search.
        self.gridSearch = GridSearchCV(self.model, parameters, scoring=scorer, **kwargs)
        self.gridSearch = self.gridSearch.fit(self.xTrainingData, self.yTrainingData)

        # Set the model (classifier) to the best combination of parameters.
        self.model = self.gridSearch.best_estimator_

        # Fit the best algorithm to the data.
        self.model.fit(self.xTrainingData, self.yTrainingData)


    def DisplayChosenParameters(self, useMarkDown=False):
        """
        Prints the model performance scores.

        Parameters
        ----------
        useMarkDown : bool
            If true, markdown output is enabled.

        Returns
        -------
        None.

        """
        lendres.Console.PrintBoldMessage("Chosen Model Parameters", useMarkDown=useMarkDown)
        display(self.gridSearch.best_params_)


    def CreateCostComplexityPruningModel(self, **kwargs):
        self.model              = DecisionTreeClassifier(random_state=1, **kwargs)
        self.CostComplexityPath = self.model.cost_complexity_pruning_path(self.xTrainingData, self.yTrainingData)

        ccpAlphas                    = self.CostComplexityPath.ccp_alphas
        self.decisionTreeHelpers = []
        
        for ccpAlpha in ccpAlphas:
            decisionTreeHelper = DecisionTreeHelper.FromData(self, deep=False)
            decisionTreeHelper.CreateModel(ccp_alpha=ccpAlpha)
            decisionTreeHelper.Predict()
            self.decisionTreeHelpers.append(decisionTreeHelper)
            
            
    def CreateAlphasVersusScoresPlot(self, criteria, scale=1.0):

        criteriaName = criteria.title()
        
        # Must be run before creating figure or plotting data.
        lendres.Plotting.FormatPlot(scale=scale)
        
        axis = plt.gca()
        
        trainingScores = []
        testScores     = []
        
        for decisionTreeHelper in self.decisionTreeHelpers:
            decisionTreeHelper.Predict()
            performanceScores = decisionTreeHelper.GetModelPerformanceScores()
            trainingScores.append(performanceScores.loc["Training", criteriaName])
            testScores.append(performanceScores.loc["Testing", criteriaName])

        ccpAlphas = self.CostComplexityPath.ccp_alphas

        # Must be run before creating figure or plotting data.
        lendres.Plotting.FormatPlot(scale=scale)
        
        axis = plt.gca()
        
        axis.plot(ccpAlphas, trainingScores, marker='o', label="Training", drawstyle="steps-post", color="#1f77b4")
        axis.plot(ccpAlphas, testScores, marker='o', label="Testing", drawstyle="steps-post", color="#ff7f0e")

        axis.set(title=criteriaName+" vs Alpha", xlabel="Alpha", ylabel=criteriaName)
        axis.legend()

        plt.show()


    def CreateImpunityVersusAlphaPlot(self, scale=1.0):
        ccpAlphas  = self.CostComplexityPath.ccp_alphas[:-1]
        impurities = self.CostComplexityPath.impurities[:-1]
        
        # Must be run before creating figure or plotting data.
        lendres.Plotting.FormatPlot(scale=scale)

        axis = plt.gca()
        axis.plot(ccpAlphas, impurities, marker='o', drawstyle="steps-post")
        axis.set(title="Total Impurity vs Effective Alpha\nTraing Data", xlabel="Effective Alpha", ylabel="Total Impurity of Leaves")

        plt.show()


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

        # Must be run before creating figure or plotting data.
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