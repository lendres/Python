"""
Created on Sat Dec  4 18:49:50 2021

@author: Lance A. Endres
"""

import matplotlib.pyplot as plt
import os
import shutil

class PlotHelper():
    @classmethod
    def setUp(cls):
        # Default output directory for saving plots.
        cls.defaultOutputDirector  = ".\\Output\\"
        cls.useDefaultOutputFolder = True


    @classmethod
    def ApplyPlotToEachCategory(cls, function, data, categories, save=False):
        """
        Creates a new figure for every entry in the list of categories.
    
        Parameters
        ----------
        data : Pandas DataFrame
            The data.
        categories : an arry or list of strings
            Category names in the DataFrame.
        save : bool
            If true, the plots are saved to the default plotting directory.
    
        Returns
        -------
        None.
        """
    
        for category in categories:
            figure = function(data, category)
    
            if save:
                fileName = function.__name__ + category.title() + " Category"
                cls.SavePlot(fileName, figure=figure)


    @classmethod
    def FormatPlot(cls, scale=1.0, width=10, height=6, transparentLegend=False):
        """
        Sets the font sizes, weights, and other properties of a plot.
    
        Parameters
        ----------
        scale : float, optional
            Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the
            output scale of the plot. The default is 1.0.
       width : float, optional
           The width of the figure. The default is 10.
       height : float, optional
           The height of the figure. The default is 6.
        transparentLegend : bool
            Option to create a legend with a transparent background.
    
        Returns
        -------
        None.
        """
    
        # Standard size.
        size = 20
    
        # Standard formating parameters.
        params = {
            "figure.figsize"         : (width, height),
            "font.size"              : 1.2*scale*size,
            "font.weight"            : "bold",
            "figure.titlesize"       : 1.2*scale*size,
            "figure.titleweight"     : "bold",
            "legend.fontsize"        : 0.8*scale*size,
            "legend.title_fontsize"  : 0.85*scale*size,
            "legend.edgecolor"       : "black",
            "axes.titlesize"         : 1.1*scale*size,
            "axes.titleweight"       : "bold",
            "axes.labelweight"       : "bold",
            "axes.labelsize"         : scale*size,
            "xtick.labelsize"        : 0.80*scale*size,
            "ytick.labelsize"        : 0.80*scale*size,
            "axes.titlepad"          : 25,
            "axes.linewidth"         : 1.5*scale,               # Axis border.
            "patch.linewidth"        : 1.5*scale,               # Legend border.
            "lines.linewidth"        : 3*scale
    
        }
    
        # Parameters to create a legend with a border and transparent background.
        transparentLegendParams = {
            "legend.framealpha"      : None,
            "legend.facecolor"       : (1, 1, 1, 0)
        }
    
        # Parameters for a legend with a border and a solid background.
        nonTransparentLegendParams = {
            "legend.framealpha"      : 1.0,
            "legend.facecolor"       : 'inherit'
        }
    
        if transparentLegend:
            params.update(transparentLegendParams)
        else:
            params.update(nonTransparentLegendParams)
    
        plt.rcParams.update(params)
    

    @classmethod
    def NewTopAndBottomAxisFigure(cls, category, topPercent=0.25, scale=1.0):
        """
        Creates a new figure that has two axes, one above another.
    
        Parameters
        ----------
        category : string
            Category name in the DataFrame.
        scale : double
            Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot.
    
        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        (boxAxis, historgramAxis) : axis array
            The top axis and bottom axis, respectively, for the box plot and histogram.
        """
    
        # Check input.
        if topPercent <= 0 or topPercent >= 1.0:
            raise Exception("Top percentage out of range.")
    
        # The format setup needs to be run first.
        cls.FormatPlot(scale=scale)
    
        figure, (boxAxis, histogramAxis) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (topPercent, 1-topPercent)})
    
        figure.suptitle("\"" + category.title() + "\"" + " Category")
    
        return (figure, (boxAxis, histogramAxis))


    @classmethod
    def GetDefaultOutputDirectory(cls):
        """
        Gets the default output location for saving figures.
    
        Returns
        -------
        : string
            The default saving location for figures.
    
        """
        return os.path.join(os.getcwd(), cls.defaultOutputDirector)
    

    @classmethod
    def DeleteOutputDirectory(cls):
        """
        Removes all the files and subdirectories in the default output directory and then deletes the directory.
    
        Parameters
        ----------
        None.
    
        Returns
        -------
        None.
        """
        if os.path.isdir(cls.defaultOutputDirector):
            shutil.rmtree(cls.defaultOutputDirector)
    

    @classmethod
    def SavePlot(cls, saveFileName, figure=None, useDefaultOutputFolder=False):
        """
        Saves a plot with a set of default parameters.
    
        Parameters
        ----------
        saveFileName : string
            The (optionally) path and file name to save the image to.
        figure : Figure
            The figure to save.  If "None" is specified, the current figure will be used.
        useDefaultOutputFolder : bool
            If true, the image is saved to a subfolder or the current folder called "Output."  If false, the
            path is assumed to be part of "saveFileName."  If false and no path is part of "saveFileName" the
            current directory is used.  The default is "False."
    
        Returns
        -------
        None.
        """
    
        if figure == None:
            figure=plt.gcf()
    
        # Default is to use the save path and file name exactly as it was passed.
        path = saveFileName
    
        # If the default ouptput folder is specified, we need to make sure it exists and update
        # the save path to account for it.
        if useDefaultOutputFolder:
    
            # Directory needs to exist.
            if not os.path.isdir(cls.defaultOutputDirector):
                os.mkdir(cls.defaultOutputDirector)
    
            # Update path.
            path = cls.defaultOutputDirector + saveFileName
    
        # And, finally, get down to the work.
        figure.savefig(path, dpi=500, transparent=True, bbox_inches="tight")


# Set up the class.
PlotHelper.setUp()