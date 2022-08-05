"""
Created on Thu August  4, 2022
@author: Lance A. Endres
"""
import os
import tensorflow                                as tf

from   lendres.PlotHelper                        import PlotHelper
from   lendres.SaveHistoryCallback               import SaveHistoryCallback


class KerasTuningOutputCallback(SaveHistoryCallback):
    """
    Generates output for each trial during Keras hyperparameter tuning.
    """


    def __init__(self, tensorFlowHelper, directory, projectName, processingFunction=None):
        """
        Constructor.

        Parameters
        ----------
        tensorFlowHeler : TensorFlowHelper
            TensorFlowHelper that contains the model and history.
        directory : string
            Output directory used by the Keras tuner.  The same as "directory" argument in Tuner.search.
        projectName : string
            The project directory used by the Keras tuner.  The same as the "project_name" argument in Tuner.search.
        processingFunction : function
            A function with the signature:
                processingFunction(tensorFlowHelper, ouputDirectory)
            If a function is not specified, a default one will be used.  The processing function is passed a TensorFlowHelper
            that is ready for post processing (prediction has already been done) and the output directory to write any output to.

        Returns
        -------
        None.
        """
        super().__init__(tensorFlowHelper, saveToDisk=False)

        self.outputFolder     = os.path.join(directory, projectName)

        self.processingFunction = processingFunction or self.DefaultProcessingFunction

        # Needed so we can specify our own directories to save figures to.
        PlotHelper.useDefaultOutputFolder = False


    def on_train_end(self, logs=None):
        """
        End of training call back.

        Parameters
        ----------
        logs : dictionary
            metric results for this training epoch, and for the validation epoch if validation is performed.  Validation
            result keys are prefixed with val_. For training epoch, the values of the Model's metrics are returned.
            Example : {'loss': 0.2, 'accuracy': 0.7}.

        Returns
        -------
        None.
        """
        super().on_train_end(logs)

        # The self.model is set by the hyperparameter tuner.
        # It doesn't seem possible to call "predict" on the original model.  It erases the logs (no val_accuracy present) for
        # some reason.  Instead, we will clone the model and use that for our own prediction purposes.
        self.tensorFlowHelper.model = tf.keras.models.clone_model(self.model)
        self.tensorFlowHelper.Predict()

        # It seems getting the output directory from Keras requires subclassing the tuner.  As a shortcut, we will
        # just assme the current directory is the last one.  This is not a great method, but will function for now.
        directories = [x for x in filter(lambda x : x.is_dir(), os.scandir(self.outputFolder))]
        outputDirectory  = directories[-1].path

        self.processingFunction(self.tensorFlowHelper, outputDirectory)


    def DefaultProcessingFunction(self, tensorFlowHelper, outputDirectory):
        """
        Default processing function.  Creates loss and accuracy history plots and writes the accuracy scores to a file.

        Parameters
        ----------
        tensorFlowHeler : TensorFlowHelper
            TensorFlowHelper that contains the model and history.  It has had prediction run and is ready for output.
        outputDirectory : string
            The directory to write output files to.

        Returns
        -------
        None.
        """
        path   = os.path.join(outputDirectory, "loss.jpg")
        figure = self.tensorFlowHelper.CreateTrainingAndValidationHistoryPlot("loss");
        figure = PlotHelper.SavePlot(path, figure)

        path   = os.path.join(outputDirectory, "accuracy.jpg")
        figure = self.tensorFlowHelper.CreateTrainingAndValidationHistoryPlot("accuracy");
        figure = PlotHelper.SavePlot(path, figure)

        path   = os.path.join(outputDirectory, "performace_scores.csv")
        performanceScores = self.tensorFlowHelper.GetModelPerformanceScores(scores="Accuracy", final=False)
        performanceScores.to_csv(path)