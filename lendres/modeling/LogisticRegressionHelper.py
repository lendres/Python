"""
Created on January 19, 2022
@author: Lance A. Endres
"""
import pandas                                                        as pd
import numpy                                                         as np

from   sklearn.linear_model                                          import LogisticRegression

from   lendres.plotting.PlotMaker                                    import PlotMaker
from   lendres.modeling.CategoricalRegressionHelper                  import CategoricalRegressionHelper
from   lendres.modeling.LogisticRegressionTools                      import LogisticRegressionTools


class LogisticRegressionHelper(CategoricalRegressionHelper):

    def __init__(self, dataHelper, model=None, description=""):
        """
        Constructor.

        Parameters
        ----------
        dataHelper : DataHelper
            DataHelper that has the data in a pandas.DataFrame.
        model : Model
            A regression model.
        description : string
            A description of the model.

        Returns
        -------
        None.
        """
        if model == None:
            model = LogisticRegressionHelper.CreateDefaultModel()

        super().__init__(dataHelper, model, description)


    @classmethod
    def CreateDefaultModel(cls, **kwargs):
        """
        Creates a linear regression model.  Splits the data and creates the model.

        Parameters
        ----------
        **kwargs : keyword arguments
            These arguments are passed on to the model.

        Returns
        -------
        None.
        """
        return LogisticRegression(random_state=1, **kwargs)


    def PredictProbabilities(self):
        """
        Runs the probability prediction (model.predict_proba) on the training and test data.  The results are stored in
        the yTrainingPredicted and yTestingPredicted variables.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # Predict probabilities.  The second column (probability of success) is retained.
        # The first column (probability of not-success) is discarded.
        self.yTrainingPredicted = self.model.predict_proba(self.dataHelper.xTrainingData)[:, 1]
        self.yTestingPredicted  = self.model.predict_proba(self.dataHelper.xTestingData)[:, 1]

        if len(self.dataHelper.yValidationData) != 0:
            self.yValidationPredicted = self.model.predict_proba(self.dataHelper.xValidationData)[:, 1]


    def PredictWithThreshold(self, threshold):
        """
        Runs the probability prediction (model.predict_proba) on the training and test data.  The results are stored in
        the yTrainingPredicted and yTestingPredicted variables.

        Parameters
        ----------
        threshold : float
            Threshold for classifying the observation success.

        Returns
        -------
        None.
        """
        # Predictions from the independent variables using the model.
        self.PredictProbabilities()

        self.yTrainingPredicted = self.yTrainingPredicted > threshold
        self.yTrainingPredicted = np.round(self.yTrainingPredicted)

        self.yTestingPredicted  = self.yTestingPredicted  > threshold
        self.yTestingPredicted  = np.round(self.yTestingPredicted)

        if len(self.dataHelper.yValidationData) != 0:
            self.yValidationPredicted  = self.yValidationPredicted  > threshold
            self.yValidationPredicted  = np.round(self.yValidationPredicted)


    def GetOdds(self, sort=False):
        """
        Converts the coefficients to odds and percent changes.

        Parameters
        ----------
        sort : bool, optional
            Specifies if the results should be sorted.  Default is false.

        Returns
        -------
        dataFrame : pandas.dataFrame
            The odds and percent changes in a data frame.
        """
        odds = np.exp(self.model.coef_[0])

        # finding the percentage change
        percentChange = (odds - 1) * 100

        # Remove limit from number of columns to display.
        pd.set_option("display.max_columns", None)

        # Add the odds to a dataframe.
        dataFrame = pd.DataFrame({"Odds" : odds,
                                 "Percent Change" : percentChange},
                                 index=self.dataHelper.xTrainingData.columns)

        if sort:
            dataFrame.sort_values("Odds", axis=0, ascending=False, inplace=True)

        return dataFrame


    def CreateRocCurvePlot(self, dataSet="training", **kwargs):
        """
        Creates a plot of the receiver operatoring characteristic curve(s).

        Parameters
        ----------
        dataSet : string
            Which data set(s) to plot.
            training - Plots the results from the training data.
            testing  - Plots the results from the test data.
            both     - Plots the results from both the training and test data.
        **kwargs :  keyword arguments
            keyword arguments pass on to the plot formating function.

        Returns
        -------
        figure : matplotlib.pyplot.figure
            The newly created figure.
        axis : matplotlib.pyplot.axis
            The axis of the plot.
        """
        self.PredictProbabilities()

        # Plot the ROC curve(s).
        plottingData = {}
        if dataSet == "training" or dataSet == "both":
            plottingData["training"] = [self.dataHelper.yTrainingData, self.yTrainingPredicted]

        if dataSet == "testing" or dataSet == "both":
            plottingData["testing"] = [self.dataHelper.yTestingData, self.yTestingPredicted]

        figure, axis = LogisticRegressionTools.CreateRocCurvePlot(plottingData, "Logistic Regression")

        return figure, axis