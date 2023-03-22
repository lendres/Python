"""
Created on December 4, 2021
@author: Lance A. Endres
"""
import numpy                                     as np

import matplotlib
import matplotlib.pyplot                         as plt

import seaborn                                   as sns

import os
import shutil
from   io                                        import BytesIO
import base64
from   PIL                                       import Image
from   PIL                                       import ImageChops


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
            cls.ResetMatPlotLib()
        elif formatStyle == "seaborn":
            cls.UseSeabornColorCodes()


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
        (leftAxis, rightAxis1, rightAxis2, ..., rightAxisN) : axis list
            The left axis and all the right axes.
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
        cls.SetZOrderOfMultipleAxisFigure(axes)

        return (figure, axes)


    @classmethod
    def SetZOrderOfMultipleAxisFigure(cls, axes):
        """
        Puts the right hand axis of a two axis plot

        Parameters
        ----------
        axes : axis
            The axes.  The axis with a y-axis on the left is in axes[0].  The axes with the y-axis on
            the right are in axes[0] ... axes[N].

        Returns
        -------
        None.
        """
        # This is necessary to have the axis with the left y-axis show in front of the axis with the right y-axis.
        # In order to do this, two things are required:
        #    1) Reverse the z order so that the left axis is drawn above (after) the right axis.
        #    2) Reverse the patch (background) transparency.  The patch of the axis in front (left) has to be
        #       transparent.  We want the patch of the axis in back to be the same as before, so the alpha has
        #       to be taken from the left and set on the right.
        # We use axes[-1] because it is the last axis on the right side and should be the highest in the order.  This
        # is an assumption.  The safer thing to do would be to loop through them all and retrieve the highest z-order.
        zOrderSave = axes[-1].get_zorder()

        # It seems that the right axis can have an alpha of "None" and be transparent, but if we set that on
        # the left axis, it does not produce the same result.  Therefore, if it is "None", we default to
        # completely transparent.
        alphaSave  = axes[1].patch.get_alpha()
        alphaSave  = 0 if alphaSave is None else alphaSave

        for i in range(1, len(axes)):
            axes[i].set_zorder(axes[0].get_zorder()+i)
            axes[i].patch.set_alpha(axes[0].patch.get_alpha())

        # The z orders could have been the same, in which case the first created is on top.  We need to add
        # one to make sure the left is on top.
        axes[0].set_zorder(zOrderSave+1)
        axes[0].patch.set_alpha(alphaSave)


    @classmethod
    def AlignXAxes(cls, axes, numberOfTicks=None):
        """
        Align the ticks (grid lines) of multiple x axes.  A new set of tick marks is computed
        as a linear interpretation of the existing range.  The number of tick marks is the
        same for both axes.  By setting them both to the same number of tick marks (same
        spacing between marks), the grid lines are aligned.

        Parameters
        ----------
        axes : list
            list of axes objects whose yaxis ticks are to be aligned.

        numberOfTicks : None or integer
            The number of ticks to use on the axes.  If None, the number of ticks on the
            first axis is used.

        Returns
        -------
        tickSets : list
            A list of new ticks for each axis in axis.
        """
        cls.AlignAxes(axes, "x", numberOfTicks)


    @classmethod
    def AlignYAxes(cls, axes, numberOfTicks=None):
        """
        Align the ticks (grid lines) of multiple y axes.  A new set of tick marks is computed
        as a linear interpretation of the existing range.  The number of tick marks is the
        same for both axes.  By setting them both to the same number of tick marks (same
        spacing between marks), the grid lines are aligned.

        Parameters
        ----------
        axes : list
            list of axes objects whose yaxis ticks are to be aligned.

        numberOfTicks : None or integer
            The number of ticks to use on the axes.  If None, the number of ticks on the
            first axis is used.

        Returns
        -------
        tickSets : list
            A list of new ticks for each axis in axis.
        """
        cls.AlignAxes(axes, "y", numberOfTicks)

    @classmethod
    def AlignAxes(cls, axes, which, numberOfTicks=None):
        """
        Align the ticks (grid lines) of multiple y axes.  A new set of tick marks is computed
        as a linear interpretation of the existing range.  The number of tick marks is the
        same for both axes.  By setting them both to the same number of tick marks (same
        spacing between marks), the grid lines are aligned.

        Parameters
        ----------
        axes : list
            list of axes objects whose yaxis ticks are to be aligned.
        which : string
            Which set of axes to align.  Options are "x" or "y".

        numberOfTicks : None or integer
            The number of ticks to use on the axes.  If None, the number of ticks on the
            first axis is used.

        Returns
        -------
        tickSets : list
            A list of new ticks for each axis in axis.
        """
        if which == "x":
            tickSets = [axis.get_xticks() for axis in axes]
        elif which == "y":
            tickSets = [axis.get_yticks() for axis in axes]
        else:
            raise Exception("Invalid direction specified in \"AlignAxes\"")

        # If the number of ticks was not specified, use the number of ticks on the first axis.
        if numberOfTicks is None:
            numberOfTicks = len(tickSets[0])

        numberOfIntervals = numberOfTicks - 1

        # The first axis is remains the same.  Those ticks should already be nicely spaced.
        tickSets[0] = np.linspace(tickSets[0][0], tickSets[0][-1], numberOfTicks, endpoint=True)

        #####
        # This method needs to be adjusted to account for different scale.  E.g. 0.2-0.8 versus 20-80.
        #####
        # Create a new set of tick marks that have the same number of ticks for each axis.
        # We have to scale the interval between tick marks.  We want them to be nice numbers (not something
        # like 72.2351).  To do this, we calculate a new interval by rounding up the existing spacing.  Rounding
        # up ensures no plotted data is cut off by scaling it down slightly.
        for i in range(1, len(tickSets)):
            span     = (tickSets[i][-1] - tickSets[i][0])
            interval = np.ceil(span / numberOfIntervals)
            tickSets[i] = np.linspace(tickSets[i][0], tickSets[i][0]+interval*numberOfIntervals, numberOfTicks, endpoint=True)

        # Set ticks for each axis.
        if which == "x":
            for axis, tickSet in zip(axes, tickSets):
                axis.set(xticks=tickSet, xlim=(tickSet[0], tickSet[-1]))
        elif which == "y":
            for axis, tickSet in zip(axes, tickSets):
                axis.set(yticks=tickSet, ylim=(tickSet[0], tickSet[-1]))

        return tickSets


    @classmethod
    def ReverseYAxisLimits(cls, axis):
        """
        Switches the upper and lower limits so that the highest value is on top.

        Parameters
        ----------
        axis : matplotlib.axes.Axes
            Axis change the limits on.

        Returns
        -------
        None.
        """
        tickSet = axis.get_yticks()
        axis.set_ylim((tickSet[-1], tickSet[0]))


    @classmethod
    def SetXAxisLimits(cls, axis, limits, numberOfTicks=None):
        """
        Sets the x-axis limits.  Allows specifying the number of ticks to use.

        Parameters
        ----------
        axis : matplotlib.axes.Axes
            Axis change the limits on.
        limits : array like of two values
            The lower and upper limits of the axis.
        numberOfTicks : int, optional
            The number of ticks (labeled points) to show. The default is None.

        Returns
        -------
        None.
        """
        tickSet       = axis.get_xticks()

        if numberOfTicks is None:
            numberOfTicks = len(tickSet)

        tickSet = np.linspace(limits[0], limits[-1], numberOfTicks, endpoint=True)
        axis.set_xticks(tickSet)
        axis.set_xlim((tickSet[0], tickSet[-1]))


    @classmethod
    def SetYAxisLimits(cls, axis, limits, numberOfTicks=None):
        """
        Sets the y-axis limits.  Allows specifying the number of ticks to use.

        Parameters
        ----------
        axis : matplotlib.axes.Axes
            Axis change the limits on.
        limits : array like of two values
            The lower and upper limits of the axis.
        numberOfTicks : int, optional
            The number of ticks (labeled points) to show. The default is None.

        Returns
        -------
        None.
        """
        tickSet       = axis.get_yticks()

        if numberOfTicks is None:
            numberOfTicks = len(tickSet)

        tickSet = np.linspace(limits[0], limits[-1], numberOfTicks, endpoint=True)
        axis.set_yticks(tickSet)
        axis.set_ylim((tickSet[0], tickSet[-1]))


    @classmethod
    def GetYBoundaries(cls, axis):
        """
        Gets the minimum and maximum Y tick marks on the axis.

        Parameters
        ----------
        axis : axis
            Axis to extract the information from.

        Returns
        -------
        yBoundries : list
            The minimim and maximum tick mark as a list.
        """
        ticks      = axis.get_yticks()
        yBoundries = [ticks[0], ticks[-1]]
        return yBoundries


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
    def GetdefaultOutputDirectoryy(cls):
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