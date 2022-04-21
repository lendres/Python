"""
Created on December 4, 2021
@author: Lance A. Endres
"""
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

import os
import shutil

class PlotHelper():
    # Class level variables.

    # Default location of saved files is a subfolder of the current working directory.
    defaultOutputDirector  = ".\\Output\\"

    #If true, the image is saved to a subfolder or the current folder called "Output."  If false, the path is assumed to be part
    # of "saveFileName."  If false and no path is part of "saveFileName" the current directory is used.
    useDefaultOutputFolder = True

    # Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot. The default is 1.0.
    scale                  = 1.0

    @classmethod
    def ApplyPlotToEachCategory(cls, data, columns, plotFunction, save=False, **kwargs):
        """
        Creates a new figure for every entry in the list of columns.

        Parameters
        ----------
        data : Pandas DataFrame
            The data.
        columns : an arry or list of strings
            Column names in the DataFrame.
        plotFunction : function
            Plotting function to apply to all columns.
        save : bool
            If true, the plots are saved to the default plotting directory.
        **kwargs : keyword arguments
            These arguments are passed on to the plotFunction.

        Returns
        -------
        None.
        """
        for column in columns:
            figure = plotFunction(data, column, **kwargs)

            if save:
                fileName = plotFunction.__name__ + column.title() + " Category"
                cls.SavePlot(fileName, figure=figure)


    @classmethod
    def FormatPlot(cls, width=10, height=6, transparentLegend=False):
        """
        Sets the font sizes, weights, and other properties of a plot.

        Parameters
        ----------
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
            "font.size"              : 1.2*cls.scale*size,
            "font.weight"            : "bold",
            "figure.titlesize"       : 1.2*cls.scale*size,
            "figure.titleweight"     : "bold",
            "legend.fontsize"        : 0.8*cls.scale*size,
            "legend.title_fontsize"  : 0.85*cls.scale*size,
            "legend.edgecolor"       : "black",
            "axes.titlesize"         : 1.1*cls.scale*size,
            "axes.titleweight"       : "bold",
            "axes.labelweight"       : "bold",
            "axes.labelsize"         : cls.scale*size,
            "xtick.labelsize"        : 0.80*cls.scale*size,
            "ytick.labelsize"        : 0.80*cls.scale*size,
            "axes.titlepad"          : 25,
            "axes.linewidth"         : 0.75*cls.scale,               # Axis border.
            "axes.edgecolor"         : "black",                     # Axis border.
            "patch.linewidth"        : 1.5*cls.scale,               # Legend border.
            "lines.linewidth"        : 3*cls.scale

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
    def Label(cls, instance, title, xLabel, yLabel="", titlePrefix=None):
        """
        Parameters
        ----------
        instance : figure or axis
            The object to label.
        title : TYPE
            Main plot title.
        xLabel : string
            X axis label.
        yLabel : string, optional
            Y axis label.  Default is a blank string.
        titlePrefix : string or None, optional
            If supplied, the string is prepended to the title.  Default is none.

        Returns
        -------
        None.
        """
        # Create the title.
        if titlePrefix != None:
            title = titlePrefix + "\n" + title

        instance.set(title=title, ylabel=yLabel, xlabel=xLabel)


    @classmethod
    def NewTopAndBottomAxisFigure(cls, column, topPercent=0.25):
        """
        Creates a new figure that has two axes, one above another.

        Parameters
        ----------
        column : string
            Category name in the DataFrame.

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
        cls.FormatPlot()

        figure, (boxAxis, histogramAxis) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (topPercent, 1-topPercent)})

        figure.suptitle("\"" + column.title() + "\"" + " Category")

        return (figure, (boxAxis, histogramAxis))


    @classmethod
    def NewSideBySideAxisFigure(cls, title):
        """
        Creates a new figure that has two axes, one above another.

        Parameters
        ----------
        title : string
            Title to use for the plot.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        (leftAxis, rightAxis) : axis array
            The left axis and right axis, respectively.
        """
        # The format setup needs to be run first.
        cls.FormatPlot(width=15, height=5)

        figure, (leftAxis, rightAxis) = plt.subplots(1, 2)

        figure.suptitle("\"" + title.title() + "\"" + " Category")

        return (figure, (leftAxis, rightAxis))


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
    def SavePlot(cls, saveFileName, figure=None):
        """
        Saves a plot with a set of default parameters.

        Parameters
        ----------
        saveFileName : string
            The (optionally) path and file name to save the image to.
        figure : Figure
            The figure to save.  If "None" is specified, the current figure will be used.

        Returns
        -------
        None.
        """
        if figure == None:
            figure = plt.gcf()

        # Default is to use the save path and file name exactly as it was passed.
        path = saveFileName

        # If the default ouptput folder is specified, we need to make sure it exists and update
        # the save path to account for it.
        if cls.useDefaultOutputFolder:

            # Directory needs to exist.
            if not os.path.isdir(cls.defaultOutputDirector):
                os.mkdir(cls.defaultOutputDirector)

            # Update path.
            path = cls.defaultOutputDirector + saveFileName

        # And, finally, get down to the work.
        figure.savefig(path, dpi=500, transparent=True, bbox_inches="tight")