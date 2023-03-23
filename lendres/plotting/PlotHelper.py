"""
Created on December 4, 2021
@author: Lance A. Endres
"""
import matplotlib
import matplotlib.pyplot                         as plt

import seaborn                                   as sns

import os
import shutil
from   io                                        import BytesIO
import base64
from   PIL                                       import Image
from   PIL                                       import ImageChops

from   lendres.plotting.AxesHelper               import AxesHelper


class PlotHelper():
    # Class level variables.

    # Default location of saved files is a subfolder of the current working directory.
    defaultOutputDirectory      = "./Output/"

    #If true, the image is saved to a subfolder or the current folder called "Output."  If false, the path is assumed to be part
    # of "saveFileName."  If false and no path is part of "saveFileName" the current directory is used.
    usedefaultOutputDirectoryy   = True

    # Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot. The default is 1.0.
    scale                       = 1.0
    annotationScale             = 1.0

    # Used to scale the output width and height for all plots.
    # Set individual plot sizes to be set with FormatPlot(width, height).  Then, if all plots need to be resized (for example, to
    # shrink them in Jupyter Notebook), set these parameters before plotting.
    widthScale                  = 1.0
    heightScale                 = 1.0

    # Standard size.
    size                        = 20

    # Format style.  This is the default, it can be overridden in the call to "Format".
    formatStyle                 = "pyplot"
    colorStyle                  = "seaborn"


    @classmethod
    def PushSettings(cls, formatSettings):
        pass


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
        : double
            Scaled annotation size.
        """
        return cls.scale*cls.annotationScale*15


    @classmethod
    def FormatPlot(cls, formatStyle=None, width=10, height=6, transparentLegend=False):
        """
        Sets the font sizes, weights, and other properties of a plot.

        Parameters
        ----------
        formatStyle : string or None
            Specifies initial formating style.
                None : no initial formating is done.
                pyplot : resets matplotlib.pyplot to the default settings.
                seaborn : uses seaborn color codes.
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
        cls.ResetMatPlotLib()
        cls._SetFormatStyle(formatStyle)

        standardSize = cls.GetScaledStandardSize()

        # Standard formating parameters.
        parameters = {
            "figure.figsize"         : (width*PlotHelper.widthScale, height*PlotHelper.heightScale),
            "figure.dpi"             : 300,
            "font.size"              : 1.0*standardSize,
            "font.weight"            : "bold",
            "figure.titlesize"       : 1.2*standardSize,
            "figure.titleweight"     : "bold",
            "legend.fontsize"        : 0.8*standardSize,
            "legend.title_fontsize"  : 0.85*standardSize,
            "legend.edgecolor"       : "black",
            "axes.titlesize"         : 1.2*standardSize,
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
    def _SetFormatStyle(cls, formatStyle=None):
        """
        Sets the formatting style used for the plots.  For example, this can be pyplot formatting or Seaborn plotting.

        Parameters
        ----------
        formatStyle : string
            The formatting style to use.

        Returns
        -------
        None.
        """
        if formatStyle is None:
            formatStyle = cls.formatStyle

        if formatStyle == "pyplot":
            pass
        elif formatStyle == "seaborn":
            cls.UseSeabornColorCodes()


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

        figure, (boxAxis, histogramAxis) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios" : (topPercent, 1-topPercent)})

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
    def NewMultiXAxesFigure(cls, numberOfAxes):
        """
        Creates a new figure that has multiple axes that are on top of each other.  The
        axes have an aligned (shared) y-axis.

        Parameters
        ----------
        numberOfAxes : int
            The number of axes to create.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        (axis1, axis2, ..., axisN) : axis list
            The axes.
        """
        # The format setup needs to be run first.
        cls.FormatPlot()

        figure = plt.figure()
        axes   = [figure.gca()]

        for i in range(1, numberOfAxes):
            axes.append(axes[0].twiny())

            # Ideally, we would calculate an offset based on all the text sizes and spacing, but that seems challenging.
            # axes[i].xaxis.label.get_size()
            # plt.rcParams["axes.titlesize"]  plt.rcParams["axes.labelsize"] plt.rcParams["xtick.labelsize"]
            # Instead, we will use a linear scaling with a y-intercept that doesn't pass through zero.  This seems to work reasonable well.
            s1     = 55                     # First point selected at a plot scale of 1.0.  This is the size in points.
            s2     = 25                     # Second point selected at a plot scale of 0.25.  This is the size in points.
            m      = 4/3.0*(s1-s2)          # Slope.
            y0     = (4.0*s2-s1) / 3.0      # Y-intercept.
            offset = m * PlotHelper.scale + y0
            axes[i].spines["top"].set_position(("outward", offset))

        # Move the first axis ticks and label to the top.
        axes[0].xaxis.tick_top()
        axes[0].xaxis.set_label_position('top')

        return (figure, axes)


    @classmethod
    def NewMultiYAxesFigure(cls, numberOfAxes):
        """
        Creates a new figure that has multiple axes that are on top of each other.  The
        axes have an aligned (shared) x-axis.

        The first axis will be the left axis.  The remaining axes are stacked on the right side.

        Parameters
        ----------
        numberOfAxes : int
            The number of axes to create.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        (leftAxes, rightAxes1, rightAxes2, ..., rightAxesN) : axes list
            The left axes and all the right axeses.
        """
        # The format setup needs to be run first.
        cls.FormatPlot()

        figure = plt.figure()
        axes   = [figure.gca()]

        for i in range(1, numberOfAxes):
            axes.append(axes[0].twinx())
            offset = 1.0 + (i-1)*0.1
            axes[i].spines["right"].set_position(("axes", offset))

        # Change the drawing order of axes so the first one created is on top.
        AxesHelper.SetZOrderOfMultipleAxisFigure(axes)

        return (figure, axes)


    @classmethod
    def GetColorCycle(cls, colorStyle=None):
        """
        Gets the default Matplotlib colors in the color cycle.

        Parameters
        ----------

        Returns
        -------
        : list
            Colors in the color cycle.

        """
        if colorStyle is None:
            colorStyle = cls.colorStyle

        if colorStyle == "pyplot":
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors     = prop_cycle.by_key()['color']
        elif colorStyle == "seaborn":
            colors = [(0.2980392156862745,  0.4470588235294118,  0.6901960784313725),
                      (0.8666666666666667,  0.5176470588235295,  0.3215686274509804),
                      (0.3333333333333333,  0.6588235294117647,  0.40784313725490196),
                      (0.7686274509803922,  0.3058823529411765,  0.3215686274509804),
                      (0.5058823529411764,  0.4470588235294118,  0.7019607843137254),
                      (0.5764705882352941,  0.47058823529411764, 0.3764705882352941),
                      (0.8549019607843137,  0.5450980392156862,  0.7647058823529411),
                      (0.5490196078431373,  0.5490196078431373,  0.5490196078431373),
                      (0.8,                 0.7254901960784313,  0.4549019607843137),
                      (0.39215686274509803, 0.7098039215686275,  0.803921568627451)]
            #colors = sns.color_palette()

        else:
            raise Exception("Unkown color style requested.\nRequested style: "+colorStyle)

        return colors


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
        return os.path.join(os.getcwd(), cls.defaultOutputDirectory)


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
        if os.path.isdir(cls.defaultOutputDirectory):
            shutil.rmtree(cls.defaultOutputDirectory)


    @classmethod
    def SavePlot(cls, saveFileName, figure=None, transparent=True):
        """
        Saves a plot with a set of default parameters.

        Parameters
        ----------
        saveFileName : string
            The (optionally) path and file name to save the image to.
        figure : Figure
            The figure to save.  If "None" is specified, the current figure will be used.
        transparent : bool
            Specificies if the background of the plot should be transparent.

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
        if cls.usedefaultOutputDirectoryy:

            # Directory needs to exist.
            if not os.path.isdir(cls.defaultOutputDirectory):
                os.mkdir(cls.defaultOutputDirectory)

            # Update path.
            path = os.path.join(cls.defaultOutputDirectory, saveFileName)

        # And, finally, get down to the work.
        figure.savefig(path, dpi=500, transparent=transparent, bbox_inches="tight")


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
    def UseSeabornColorCodes(cls):
        """
        Uses Seaborn to create an alternate formatting.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        sns.set(color_codes=True)


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