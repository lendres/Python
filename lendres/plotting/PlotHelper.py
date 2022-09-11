"""
Created on December 4, 2021
@author: Lance A. Endres
"""
import matplotlib
import matplotlib.pyplot                         as plt

import seaborn                                   as sns
sns.set(color_codes=True)

import os
import shutil
from   io                                        import BytesIO
import base64
from   PIL                                       import Image
from   PIL                                       import ImageChops


class PlotHelper():
    # Class level variables.

    # Default location of saved files is a subfolder of the current working directory.
    defaultOutputDirector  = "./Output/"

    #If true, the image is saved to a subfolder or the current folder called "Output."  If false, the path is assumed to be part
    # of "saveFileName."  If false and no path is part of "saveFileName" the current directory is used.
    useDefaultOutputFolder = True

    # Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot. The default is 1.0.
    scale                  = 1.0
    annotationScale        = 1.0

    # Standard size.
    size                   = 20


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
    def GetScaledStandardSize(cls):
        """
        Gets the standard font size adjusted with the scaling factor.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        return cls.scale*cls.size


    @classmethod
    def GetScaledAnnotationSize(cls):
        """
        Gets the annotation font size adjusted with the scaling factor.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        return cls.scale*cls.annotationScale*15


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
        standardSize = cls.GetScaledStandardSize()

        # Standard formating parameters.
        parameters = {
            "figure.figsize"         : (width, height),
            "font.size"              : 1.2*standardSize,
            "font.weight"            : "bold",
            "figure.titlesize"       : 1.2*standardSize,
            "figure.titleweight"     : "bold",
            "legend.fontsize"        : 0.8*standardSize,
            "legend.title_fontsize"  : 0.85*standardSize,
            "legend.edgecolor"       : "black",
            "axes.titlesize"         : 1.1*standardSize,
            "axes.titleweight"       : "bold",
            "axes.labelweight"       : "bold",
            "axes.labelsize"         : standardSize,
            "xtick.labelsize"        : 0.80*standardSize,
            "ytick.labelsize"        : 0.80*standardSize,
            "axes.titlepad"          : 25,
            "axes.linewidth"         : 0.75*cls.scale,              # Axis border.
            "axes.edgecolor"         : "black",                     # Axis border.
            "patch.linewidth"        : 1.5*cls.scale,               # Legend border.
            "lines.linewidth"        : 3*cls.scale,
            "lines.markersize"       : 10*cls.scale
        }

        # Parameters to create a legend with a border and transparent background.
        transparentLegendParameters = {
            "legend.framealpha"      : None,
            "legend.facecolor"       : (1, 1, 1, 0)
        }

        # Parameters for a legend with a border and a solid background.
        nonTransparentLegendParameters = {
            "legend.framealpha"      : 1.0,
            "legend.facecolor"       : 'inherit'
        }

        if transparentLegend:
            parameters.update(transparentLegendParameters)
        else:
            parameters.update(nonTransparentLegendParameters)

        plt.rcParams.update(parameters)


    @classmethod
    def Label(cls, instance, title, xLabel, yLabel="", titlePrefix=None):
        """
        Add title, x axis label, and y axis label.

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

        instance.set(title=title, xlabel=xLabel, ylabel=yLabel)


    @classmethod
    def RotateXLabels(cls, xLabelRotation):
        """
        Rotate the x-axis labels.

        Parameters
        ----------
        xLabelRotation : float
            Rotation of x labels.  If none is passed, nothing is done.

        Returns
        -------
        None.
        """
        # Option to rotate the x axis labels.
        if xLabelRotation is not None:
            plt.xticks(rotation=xLabelRotation, ha="right")



    @classmethod
    def NewTopAndBottomAxisFigure(cls, title, topPercent=0.25):
        """
        Creates a new figure that has two axes, one above another.

        Parameters
        ----------
        title : string
            Figure title.

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

        figure.suptitle(title)

        return (figure, (boxAxis, histogramAxis))


    @classmethod
    def NewSideBySideAxisFigure(cls, title, width=15, height=5):
        """
        Creates a new figure that has two axes, one above another.

        Parameters
        ----------
        title : string
            Title to use for the plot.
       width : float, optional
           The width of the figure. The default is 15.
       height : float, optional
           The height of the figure. The default is 5.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        (leftAxis, rightAxis) : axis array
            The left axis and right axis, respectively.
        """
        # The format setup needs to be run first.
        cls.FormatPlot(width=width, height=height)

        figure, (leftAxis, rightAxis) = plt.subplots(1, 2)

        figure.suptitle(title)

        return (figure, (leftAxis, rightAxis))


    @classmethod
    def SetAxisToSquare(cls, axis):
        """
        Sets the axis to have a square aspect ratio.

        Parameters
        ----------
        axis : axis
            Axis to set the aspect ratio of.

        Returns
        -------
        None.
        """
        axis.set_aspect(1./axis.get_data_ratio())


    @classmethod
    def CategoryTitle(cls, categoryName):
        """
        Formats a string as a category title.  It is converted to title case, quotes
        added around the category name and "Category" added as a suffix.

        Parameters
        ----------
        categoryName : string
            Category name to convert to a title.

        Returns
        -------
        title : string
            The category converted to a title.
        """
        return "\"" + categoryName.title() + "\"" + " Category"


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
            path = os.path.join(cls.defaultOutputDirector, saveFileName)

        # And, finally, get down to the work.
        figure.savefig(path, dpi=500, transparent=True, bbox_inches="tight")


    @classmethod
    def ResetMatPlotLib(cls):
        """
        Resets Matplotlib to the default settings.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        plt.rcParams.update(plt.rcParamsDefault)


    @classmethod
    def SavePlotToBuffer(cls, figure=None, format="png", autoCrop=False, borderSize=0):
        """
        Saves a plot to a buffer.

        Parameters
        ----------
        figure : Figure, optional
            The figure to save.  If "None" is specified, the current figure will be used.
        format : string, optional
            The image output format.  Default is "png".

        Returns
        -------
        BytesIO
            Buffer with the figure written to it.
        """
        if figure == None:
            figure = plt.gcf()

        buffer = PlotHelper.SaveToBuffer(figure, "PNG" if autoCrop else format)

        if autoCrop:
            figure = Image.open(buffer).convert("RGB")
            buffer.close()
            figure = PlotHelper.CropWhiteSpace(figure, borderSize)
            buffer = PlotHelper.SaveToBuffer(figure, format)

        image     = buffer.getvalue()
        plot      = base64.b64encode(image)
        plot      = plot.decode("utf-8")

        buffer.close()

        return plot


    @classmethod
    def SaveToBuffer(cls, figure, format="PNG"):
        """
        Saves a figure or image to an IO byte buffer.

        Parameters
        ----------
        figure : matplotlib.figure.Figure or PIL.Image.Image, optional
            The figure/image to save.
        format : string, optional
            The image output format.  Default is "png".

        Returns
        -------
        BytesIO
        """
        buffer = BytesIO()

        figureType = type(figure)

        if figureType == matplotlib.figure.Figure:
            figure.savefig(buffer, format=format, bbox_inches="tight")
        elif figureType == Image.Image:
            figure.save(buffer, format=format, bbox_inches="tight")
        else:
            raise Exception("Unknown figure type.")

        buffer.seek(0)
        return buffer


    @classmethod
    def CropWhiteSpace(cls, image, borderSize):
        """
        Crops white space from the border of an image.

        Parameters
        ----------
        image : ByteIO
            An image to crop the border.
        borderSize : int
            The size of the border, in pixels, to leave remaing around the edge.

        Returns
        -------
        BytesIO
        """
        backGround = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        difference = ImageChops.difference(backGround, image)

        #difference = ImageChops.add(difference, difference, 2.0, -100)

        boundingBox = difference.getbbox()

        if boundingBox:
            boundingBox = [
                boundingBox[0]-borderSize,
                boundingBox[1]-borderSize,
                boundingBox[2]+borderSize,
                boundingBox[3]+borderSize
            ]
            return image.crop(boundingBox)
        else:
            return image