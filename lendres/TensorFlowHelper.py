"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import numpy                    as np
import matplotlib.pyplot        as plt

import seaborn                  as sns
sns.set(color_codes=True)


from lendres.PlotHelper         import PlotHelper

class TensorFlowHelper():
    # Class level variables.


    @classmethod
    def CreateTrainingAndValidationHistoryPlot(history, parameter):
        """
        Plots the confusion matrix for the model output.

        Parameters
        ----------
        history : keras.callbacks.History
        parameter : string
            The parameter to plot.

        Returns
        -------
        figure : Matplotlib.Figure
        """
        # Must be called first.
        PlotHelper.FormatPlot()

        plt.plot(history.history[parameter])
        plt.plot(history.history["val_"+parameter])

        # Create titles and set legend.
        plt.gca.set(title="Model "+parameter.title(), xlabel="Epoch", ylabel=parameter.title())
        plt.legend(["Training", "Validation"], loc="best")

        figure = plt.gcf()

        plt.show()

        return figure