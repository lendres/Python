"""
Created on January 19, 2022.
@author: Lance A. Endres
"""
import pandas                                    as pd
import numpy                                     as np
import seaborn                                   as sns
from   sklearn                                   import metrics

from   matplotlib                                import pyplot                     as plt

from   lendres.ConsoleHelper                     import ConsoleHelper
from   lendres.PlotHelper                        import PlotHelper


class ModelHelper:

    savedModelHelpers         = {}

    # Used to alter how the recall score is calculated for model comparisons.  If doing a
    # non-binary classification change to "micro," for example.
    scoringAverage             = "binary"


    def __init__(self, dataHelper, model, description=""):
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
        self.dataHelper                = dataHelper

        self.yTrainingPredicted        = []
        self.yValidationPredicted      = []
        self.yTestingPredicted         = []

        self.model                     = model

        # Features for comparing models.
        self.description               = description


    def CopyData(self, original, deep=False):
        """
        Copies the data from another ModelHelper.  Does not copy any models built
        or output produced.

        Parameters
        ----------
        original : ModelHelper
            The source of the data.
        deep : bool, optional
            Specifies if a deep copy should be done. The default is False.

        Returns
        -------
        None.
        """
        self.dataHelper                = original.dataHelper.Copy()


    @classmethod
    def RunModels(cls, modelHelpers):
        """
        Runs all the models.  The model is saved for later conparison.

        Parameters
        ----------
        modelHelpers : List of ModelHelper
            List of models to run.

        Returns
        -------
        None.
        """
        for modelHelper in modelHelpers:
            modelHelper.dataHelper.consoleHelper.PrintSectionTitle(modelHelper.description)
            ModelHelper.RunModel(modelHelper)


    @classmethod
    def RunModel(cls, regressionHelper, saveModel=True, **kwargs):
        """
        Runs one model, plots the testing confusion matrix, and displays the performance scores.

        Parameters
        ----------
        regressionHelper : ModelHelper
            Model to run.
        saveModel : bool
            If true, the model is saved for later comparison.

        Returns
        -------
        None.
        """
        # Store to use for comparison.
        if saveModel:
            ModelHelper.SaveModelHelper(regressionHelper)

        # Create the model.
        regressionHelper.CreateModel(**kwargs)
        regressionHelper.Predict()

        # Output.
        regressionHelper.CreateConfusionMatrixPlot(dataSet="testing")
        regressionHelper.DisplayModelPerformanceScores(final=False)


    @classmethod
    def SaveModelHelper(cls, modelHelper):
        """
        Saves the ModelHelper in a dictionary.  The ModelHelpers are indexed by their name as defined
        by the GetName function.

        ModelHelpers with the same name cannot be saved, they will over write each other.  This is
        by design, so that a test can be run mulple time and be tuned while ModelHelper only saves
        the last version passed to it.

        Parameters
        ----------
        modelHelper : ModelHelper
            ModelHelper to save.

        Returns
        -------
        None.
        """
        cls.savedModelHelpers[modelHelper.GetName()] = modelHelper


    @classmethod
    def GetModelComparisons(cls, scores, modelHelpers=None):
        """
        Creates a comparison of a score across several models.

        Extracts the training and testing "score" for each model and puts it into a DataFrame.  The models must
        all have a "GetModelPerformanceScores" function that returns a DataFrame that contains a "score" column
        and "Testing" and "Training" rows.

        If the model has a model.description and it is not blank, that value is used for the name of the model,
        otherwise, the class name of the model is used.

        Parameters
        ----------
        modelHelpers : list or dictionary
            List of ModelHelpers to compare.  If none is supplied, the list is taken from those stored in
            the ModelHelper.
        scores : string or list of strings
            Score to extract from the list of performance scores.  This must be a column in the DataFrame
            returned by the "GetModelPerformanceScores" function.

        Returns
        -------
        comparisonFrame : pandas.DataFrame
            A DataFrame that contains a list of the models and each models training and testing score.
        """
        # Ensure scores is a list.
        if type(scores) != list:
            scores = [scores]

        # Get a list of the ModelHelpers.
        if modelHelpers == None:
            modelHelpers = ModelHelper.savedModelHelpers

        # If it is a dictionary, flatten it into a list for easier use.
        if type(modelHelpers) == dict:
            modelHelpers = list(modelHelpers.values())

        # Create the index for the DataFrame.
        index   = []
        for modelHelper in modelHelpers:
            index.append(modelHelper.GetName())

        # Create the column names for the DataFrame.
        columns = []
        for score in scores:
            columns.append("Training " + score)
            if len(modelHelpers[0].dataHelper.xValidationData) != 0:
                columns.append("Validation " + score)
            columns.append("Testing " + score)

        # Initialize the DataFrame to the correct size and add the index and columns.
        comparisonFrame = pd.DataFrame(index=index, columns=columns)

        for modelHelper in modelHelpers:
            results = modelHelper.GetModelPerformanceScores(final=True)

            for score in scores:
                comparisonFrame.loc[modelHelper.GetName(), "Training "+score] = results.loc["Training", score]
                if len(modelHelper.dataHelper.xValidationData) != 0:
                    comparisonFrame.loc[modelHelper.GetName(), "Validation "+score]  = results.loc["Validation", score]
                comparisonFrame.loc[modelHelper.GetName(), "Testing "+score]  = results.loc["Testing", score]

        return comparisonFrame


    @classmethod
    def DisplayModelComparisons(cls, scores, modelHelpers=None):
        """
        Prints the model comparisons.

        Parameters
        ----------
        scores : string or list of strings.
            The score or scores to print out.
        modelHelpers : list of ModelHelpers
            A list of ModelHelpers to get and print the scores of.  If None, the saved list of ModelHelpers is used.

        Returns
        -------
        None.
        """
        comparisons = cls.GetModelComparisons(scores, modelHelpers)

        modelHelper = None
        if modelHelpers == None:
            modelHelper = list(ModelHelper.savedModelHelpers.values())[0]
        else:
            modelHelper = modelHelpers[0]

        pd.set_option("display.float_format",  "{:0.3f}".format)
        modelHelper.dataHelper.consoleHelper.Display(comparisons, ConsoleHelper.VERBOSEREQUESTED)
        pd.reset_option("display.float_format")


    @classmethod
    def CreateScorePlotForAllModels(cls, score, width=20):
        """
        Creates a bar plot of a score for all saved models.

        Parameters
        ----------
        score : string
            The scoring metric to plot.
        width : int
            The plot width.

        Returns
        -------
        None.
        """
        dataFrame = ModelHelper.GetModelComparisons(score)

        # Prepare the DataFrame for plotting.  This transforms the DataFrame
        dataFrame = dataFrame.transpose()
        dataFrame.reset_index(inplace=True)
        dataFrame = dataFrame.melt(id_vars='index')
        dataFrame.rename(columns={"index" : "Index", "variable" : "Model", "value" : score}, inplace=True)

        # Create the plot.
        PlotHelper.FormatPlot(width=width, height=4)
        axes = sns.barplot(x="Model", y=score, data=dataFrame, hue="Index")

        # Clear the title legend, move its location to top center, and list the entries horizontally.
        axes.legend().set_title(None)
        axes.legend(bbox_to_anchor=(0.5,1.2), loc="upper center", ncol=len(dataFrame.columns))

        # Rotate the X axis labels to vertical so they fit without running together.
        plt.xticks(rotation=90)

        plt.show()


    def GetName(self):
        """
        Gets the name of the model.  If a description has been provided, that is used.  Otherwise,
        the name of the class is returned.

        Returns
        -------
        : string
            A string representing the name of the model.
        """
        # The name of the model to use as the index.  Use the more useful "description"
        # if it is available, otherwise use the calls name.
        if self.description == "":
            return self.__class__.__name__
        else:
            return self.description


    def PrintClassName(self):
        """
        Displays the class name according to the behavior specified in the ConsoleHelper.

        Returns
        -------
        None.
        """
        self.dataHelper.consoleHelper.Print(self.__class__.__name__, ConsoleHelper.VERBOSEREQUESTED)


    def fit(self, x, y=None):
        """
        Fits the model.  Used to create capatibility with the sklearn API.

        Parameters
        ----------
        **kwargs : keyword arguments
            These arguments are passed on to the DecisionTreeClassifier.

        Returns
        -------
        None.
        """
        self.Fit()


    def FitPredict(self):
        """
        Fits a hyperparameter search and runs predict.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.Fit()
        self.Predict()


    def Fit(self):
        """
        Fits the model.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        if len(self.dataHelper.xTrainingData) == 0:
            raise Exception("The data has not been split.")

        if self.model == None:
            raise Exception("The model has not been created.")

        self.model.fit(self.dataHelper.xTrainingData, self.dataHelper.yTrainingData)


    def Predict(self):
        """
        Runs the prediction (model.predict) on the training and test data.  The results are stored in
        the yTrainingPredicted and yTestingPredicted variables.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # Predict on the training and testing data.
        self.yTrainingPredicted        = self.model.predict(self.dataHelper.xTrainingData)
        self.yTestingPredicted         = self.model.predict(self.dataHelper.xTestingData)

        if len(self.dataHelper.yValidationData) != 0:
            self.yValidationPredicted  = self.model.predict(self.dataHelper.xValidationData)


    def GetModelCoefficients(self):
        """
        Displays the coefficients and intercept of a linear regression model.

        Parameters
        ----------
        None.

        Returns
        -------
        DataFrame that contains the model coefficients.
        """
        # Make sure the model has been initiated and of the correct type.
        if self.model == None:
            raise Exception("The regression model has not be initiated.")

        dataFrame = pd.DataFrame(np.append(self.model.coef_, self.model.intercept_),
                                 index=self.dataHelper.xTrainingData.columns.tolist()+["Intercept"],
                                 columns=["Coefficients"])
        return dataFrame


    def GetModelPerformanceScores(self, final=False):
        """
        Calculate performance metrics.  Threshold for a positive result can be specified.

        Parameters
        ----------
        final : boolean
            If true, the testing scores are included.

        Returns
        -------
        dataFrame : DataFrame
            DataFrame that contains various performance scores for the training and test data.
        """
        # Make sure the model has been initiated and of the correct type.
        if self.model == None:
            raise Exception("The regression model has not be initiated.")

        if len(self.yTrainingPredicted) == 0:
            raise Exception("The predicted values have not been calculated.")

        # Calculate scores.
        # TRAINING.
        # Accuracy.
        accuracyScores   = [metrics.accuracy_score(self.dataHelper.yTrainingData, self.yTrainingPredicted)]
        # Recall.
        recallScores     = [metrics.recall_score(self.dataHelper.yTrainingData, self.yTrainingPredicted, average=ModelHelper.scoringAverage)]
        # Precision.
        precisionScores  = [metrics.precision_score(self.dataHelper.yTrainingData, self.yTrainingPredicted, zero_division=0, average=ModelHelper.scoringAverage)]
        # F1.
        f1Scores         = [metrics.f1_score(self.dataHelper.yTrainingData, self.yTrainingPredicted, average=ModelHelper.scoringAverage)]
        # Index.
        index            = ["Training"]

        # VALIDATION.
        if len(self.dataHelper.yValidationData) != 0:
           # Accuracy.
            accuracyScores.append(metrics.accuracy_score(self.dataHelper.yValidationData, self.yValidationPredicted))
            # Recall.
            recallScores.append(metrics.recall_score(self.dataHelper.yValidationData, self.yValidationPredicted, average=ModelHelper.scoringAverage))
            # Precision.
            precisionScores.append(metrics.precision_score(self.dataHelper.yValidationData, self.yValidationPredicted, zero_division=0, average=ModelHelper.scoringAverage))
            # F1.
            f1Scores.append(metrics.f1_score(self.dataHelper.yValidationData, self.yValidationPredicted, average=ModelHelper.scoringAverage))
            # Index.
            index.append("Validation")

        if final:
            # TESTING.
            # Accuracy.
            accuracyScores.append(metrics.accuracy_score(self.dataHelper.yTestingData, self.yTestingPredicted))
            # Recall.
            recallScores.append(metrics.recall_score(self.dataHelper.yTestingData, self.yTestingPredicted, average=ModelHelper.scoringAverage))
            # Precision.
            precisionScores.append(metrics.precision_score(self.dataHelper.yTestingData, self.yTestingPredicted, zero_division=0, average=ModelHelper.scoringAverage))
            # F1.
            f1Scores.append(metrics.f1_score(self.dataHelper.yTestingData, self.yTestingPredicted, average=ModelHelper.scoringAverage))
            # Index.
            index.append("Testing")

        # Create a DataFrame for returning the values.
        dataFrame = pd.DataFrame(
            {
                "Accuracy"  : accuracyScores,
                "Recall"    : recallScores,
                "Precision" : precisionScores,
                "F1"        : f1Scores
            },
            index=index
        )

        return dataFrame


    def DisplayModelPerformanceScores(self, scores=None, final=False):
        """
        Displays the model performance scores based on the settings in the ConsoleHelper.

        Parameters
        ----------
        scores : string or list of strings
            Score to extract from the list of performance scores.  This must be a column in the DataFrame
            returned by the "GetModelPerformanceScores" function.
        final : boolean
            If true, the testing scores are included.

        Returns
        -------
        None.
        """
        scoresDataFrame = self.GetModelPerformanceScores(final)
        if scores != None:
            # Ensure scores is a list.
            if type(scores) != list:
                scores = [scores]
            scoresDataFrame = scoresDataFrame[scores]

        self.dataHelper.consoleHelper.PrintTitle("Performance Scores", ConsoleHelper.VERBOSEREQUESTED)
        pd.set_option("display.float_format",  "{:0.3f}".format)
        self.dataHelper.consoleHelper.Display(scoresDataFrame, ConsoleHelper.VERBOSEREQUESTED)
        pd.reset_option("display.float_format")