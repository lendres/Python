"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import numpy                    as np
import matplotlib.pyplot        as plt

import seaborn                  as sns
sns.set(color_codes=True)


from lendres.PlotHelper         import PlotHelper

class PlotMaker():
    # Class level variables.

    # Color map to use for plots.
    colorMap      = None

    @classmethod
    def CreateConfusionMatrixPlot(cls, confusionMatrix, title, titlePrefix=None):
        """
        Plots the confusion matrix for the model output.

        Parameters
        ----------
        confusionMatrix : ndarray of shape (n_classes, n_classes)
        title : string
            Main title for the data.
        titlePrefix : string or None, optional
            If supplied, the string is prepended to the title.

        Returns
        -------
        confusionMatrix : ndarray of shape (n_classes, n_classes)
            The confusion matrix for the data.
        """
        numberOfCategories = confusionMatrix.shape[0]

        if numberOfCategories != confusionMatrix.shape[1]:
            raise Exception("The confusion matrix supplied is not square.")

        # The numpy array has to be set as an object type.  If set (or allowed to assume) a
        # type of "str" the entry is created only large enough for the initial string (a character
        # type is used).  It is not possible to append to it.
        labels = np.asarray(
            [
                ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / confusionMatrix.flatten().sum())]
                for item in confusionMatrix.flatten()
            ]
        ).astype("object").reshape(numberOfCategories, numberOfCategories)

        # Tack on the type labels to the numerical information.
        if numberOfCategories == 2:
            labels[0, 0] += "\nTN"
            labels[1, 0] += "\nFN\nType 2"
            labels[0, 1] += "\nFP\nType 1"
            labels[1, 1] += "\nTP"

        # Must be run before creating figure or plotting data.
        # The standard scale for this plot will be a little higher than the normal scale.
        # Not much is shown, so we can shrink the figure size.
        PlotHelper.FormatPlot(width=5.35, height=4)

        # Create plot and set the titles.
        axis = sns.heatmap(confusionMatrix, cmap=PlotMaker.colorMap, annot=labels, annot_kws={"fontsize" : 14*PlotHelper.scale}, fmt="")
        PlotHelper.Label(axis, title=title, xLabel="Predicted", yLabel="Actual", titlePrefix=titlePrefix)

        plt.show()

        return confusionMatrix