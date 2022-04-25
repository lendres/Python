"""
Created on January 19, 2022
@author: Lance
"""
import pandas                               as pd
import numpy                                as np
import matplotlib.pyplot                    as plt

from sklearn                                import metrics
from sklearn.linear_model                   import LogisticRegression

from lendres.PlotHelper                     import PlotHelper
from lendres.CategoricalRegressionHelper    import CategoricalRegressionHelper

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

        # Must be run before creating figure or plotting data.
        # The standard scale for this plot will be a little higher than the normal scale.
        #scale *= 1.5
        PlotHelper.FormatPlot(**kwargs)

        # Plot the ROC curve(s).
        if dataSet == "both":
            self.PlotRocCurve("training")
            self.PlotRocCurve("testing")
        else:
            self.PlotRocCurve(dataSet)

        # Plot the diagonal line, the wrost fit possible line.
        plt.plot([0, 1], [0, 1], "r--")

        # Formatting the axis.
        axis  = plt.gca()
        title = "Logistic Regression\nReceiver Operating Characteristic"
        axis.set(title=title, ylabel="True Positive Rate", xlabel="False Positive Rate")
        axis.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05])

        plt.legend(loc="lower right")
        plt.show()

        figure = plt.gcf()

        return figure, axis


    def PlotRocCurve(self, dataSet="training", scale=1.0):
        """
        Plots the receiver operatoring characteristic curve.

        Parameters
        ----------
        dataSet : string
            Which data set(s) to plot.
            training - Plots the results from the training data.
            testing  - Plots the results from the test data.
        scale : double
            Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot.

        Returns
        -------
        None.
        """

        # Get the confusion matrix for the correct data set.
        if dataSet == "training":
            rocScore                                          = metrics.roc_auc_score(self.dataHelper.yTrainingData, self.yTrainingPredicted)
            falsePositiveRates, truePositiveRates, thresholds = metrics.roc_curve(self.dataHelper.yTrainingData, self.yTrainingPredicted)
            color                                             = "#1f77b4"

        elif dataSet == "testing":
            rocScore                                          = metrics.roc_auc_score(self.dataHelper.yTestingData, self.yTestingPredicted)
            falsePositiveRates, truePositiveRates, thresholds = metrics.roc_curve(self.dataHelper.yTestingData, self.yTestingPredicted)
            color                                             = "#ff7f0e"

        else:
            raise Exception("Invalid data set specified.")

        plt.plot(falsePositiveRates, truePositiveRates, label=dataSet.title()+ " Data (area = %0.2f)" % rocScore, color=color)