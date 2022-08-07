"""
Created on July 10, 2022
@author: Lance A. Endres
"""
import pandas                                            as pd
from   tensorflow.keras.callbacks                        import Callback


class SaveHistoryCallback(Callback):
    """
    This class is used for history saving while training a model.

    Example uses:
        Resume Training
            By combining history saving (using saveToDisk=True) and model checkpoints, the training progress
            can be saved to disk and restored later.  The training can then be resumed or the training results
            plotted/displayed.
        Hyperparameter Tuning Output
            Other classes can subclass this to make use of history save during hyperparameter tuning.
    """


    def __init__(self, tensorFlowHelper, saveToDisk=True):
        """
        Constructor.

        Parameters
        ----------
        tensorFlowHeler : TensorFlowHelper
            TensorFlowHelper that contains the model and history.
        saveToDisk : boolean
            If true, the history is saved to the disk as well as the TensorFlowHelper.

        Returns
        -------
        None.
        """
        super().__init__()

        self.tensorFlowHelper             = tensorFlowHelper
        self.tensorFlowHelper.historyMode = "callback"

        self.saveToDisk                   = saveToDisk


    def on_epoch_end(self, epoch, logs=None):
        """
        On epoch end callback.

        Parameters
        ----------
        epoch : integer
            Index of the epoch.
        logs : dictionary
            metric results for this training epoch, and for the validation epoch if validation is performed.  Validation
            result keys are prefixed with val_. For training epoch, the values of the Model's metrics are returned.
            Example : {'loss': 0.2, 'accuracy': 0.7}.

        Returns
        -------
        None.
        """
        super().on_epoch_end(epoch, logs)

        if self.tensorFlowHelper.history is None:
            # No history exists, so establish a new one.
            self.tensorFlowHelper.history = pd.DataFrame(logs, index=[0])
        else:
            # If history already exists, we need to append to it.
            newIndex     = self.tensorFlowHelper.history.index[-1] + 1
            logDataFrame = pd.DataFrame(logs, index=[newIndex])
            self.tensorFlowHelper.history = pd.concat([self.tensorFlowHelper.history, logDataFrame], axis=0)

        # Write the data to the disk.
        if self.saveToDisk:
            self.tensorFlowHelper.SaveHistory()