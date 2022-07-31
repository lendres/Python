"""
Created on July 10, 2022
@author: Lance A. Endres
"""
import pandas                                            as pd
from   tensorflow.keras.callbacks                        import Callback


class SaveHistoryCallback(Callback):


    def __init__(self, tensorFlowHelper):
        """
        Constructor.

        Parameters
        ----------
        tensorFlowHeler : TensorFlowHelper
            TensorFlowHelper that contains the model and history.

        Returns
        -------
        None.
        """
        super().__init__()

        self.tensorFlowHelper             = tensorFlowHelper
        self.tensorFlowHelper.historyMode = "callback"


    def on_epoch_end(self, epoch, logs=None):
        """
        Constructor.

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
        if self.tensorFlowHelper.history is None:
            # No history exists, so establish a new one.
            self.tensorFlowHelper.history = pd.DataFrame(logs, index=[0])
        else:
            # If history already exists, we need to append to it.
            newIndex     = self.tensorFlowHelper.history.index[-1] + 1
            logDataFrame = pd.DataFrame(logs, index=[newIndex])
            self.tensorFlowHelper.history = pd.concat([self.tensorFlowHelper.history, logDataFrame], axis=0)

        # Write the data to the disk.
        self.tensorFlowHelper.SaveHistory()