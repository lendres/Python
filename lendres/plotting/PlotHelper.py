"""
Created on December 4, 2021
@author: Lance A. Endres
"""
import matplotlib
import matplotlib.pyplot                                             as plt
import matplotlib.figure                                             as fig
import matplotlib.axes                                               as ax
import math

#import seaborn                                                       as sns

import os
import shutil
from   io                                                            import BytesIO
import base64
from   PIL                                                           import Image
from   PIL                                                           import ImageChops
from   PIL                                                           import ImageColor

from   lendres.plotting.FormatSettings                               import FormatSettings
from   lendres.plotting.AxesHelper                                   import AxesHelper
from   lendres.path.Path                                             import Path


class PlotHelper():
    """
    This is a helper class for creating plots.
        It does do formatting and provided formatting options.
        It does handle creating new plots for some none trivial plotting cases (e.g. multi-axes plots).
        It does have some plotting utility functions such as saving figures to files.
        It does not plot lines.

    See also FormatSettings.
    """


    # Class level variables.

    # Default location of saved files is a subfolder of the current working directory.
    defaultOutputDirectory      = "./Output/"

    #If true, the image is saved to a subfolder or the current folder called "Output."  If false, the path is assumed to be part
    # of "saveFileName."  If false and no path is part of "saveFileName" the current directory is used.
    usedefaultOutputDirectory   = True

    # Format settings.
    formatSettings              = FormatSettings()
    defaultFormatSettings       = FormatSettings()
    storedFormatSettings        = None

    currentColor                = 0


    @classmethod
    def GetSettings(cls) -> FormatSettings:
        """
        Gets the current FormatSettings.

        Returns
        -------
        : FormatSettings
            The current format settings.
        """
        return cls.formatSettings


    @classmethod
    def ResetSettings(cls):
        """
        Restores the original built-in format settings.

        Returns
        -------
        None.
        """
        cls.SetSettings(FormatSettings())


    @classmethod
    def SetSettings(cls, formatSettings:FormatSettings=None, **kwargs):
        """
        Sets the format settings.  It is necessary to supply either an instance  of FormatSettings or at least one
        keyword argument that is passed to FormatSettings.

        If keyword arguments are supplied, the original built-in setttings are used as the basis of the settings and the
        settings supplied as keyword arguments are overwritten.

        The default settings are established and any stored settings (from pushed settings) will be cleared.

        Parameters
        ----------
        formatSettings : FormatSettings, optional
            The format settings. The default is None.
        **kwargs : keyword arguments
            Keyword arguments recognized by FormatSettings.

        Returns
        -------
        None.
        """
        # If formatSettings is None, create a new instance.
        if formatSettings is None:
            formatSettings = FormatSettings(**kwargs)

        cls.formatSettings        = formatSettings
        cls.defaultFormatSettings = formatSettings
        cls.storedFormatSettings  = None


    @classmethod
    def PushSettings(cls, formatSettings:FormatSettings|str="current", **kwargs):
        """
        Sets the format settings (temporarily).  It is necessary to supply either an instance of FormatSettings or
        at least one keyword argument that is passed to FormatSettings.  The original settings are restored by
        calling "PopSettings".

        If keyword arguments are supplied, they are used to override the settings.  The value of "base" specifies if
        the "current" or "default" settings are used as the base settings to override.

        Pushing the settings does not erase or reset any parameters when keyword arguments are specified alone.  They
        key word arguments are used to overwrite/update existing values.  To reset the format settings, supply your
        own new instance of FormatSettings.

        Parameters
        ----------
        formatSettings : FormatSettings|str, optional
            Specified the basis of the settings to apply the keyword arguments to.
                "current"                 - The current format settings are used.
                "default"                 - The default format settings are used.
                FormatSettings instalnce  - The supplied instance of the format settings are used.
            The default is "current".
        **kwargs : keyword arguments
            Keyword arguments recognized by FormatSettings.

        Returns
        -------
        None.
        """
        match formatSettings:
            case "current":
                # Create a new instance by copying the existing settings.
                formatSettings = cls.formatSettings.Copy().Update(**kwargs)
            case "default":
                formatSettings = cls.defaultFormatSettings.Copy().Update(**kwargs)
            case FormatSettings():
                formatSettings = formatSettings.Copy().Update(**kwargs)
            case _:
                raise Exception("Invalid 'formatSettings' parameter provided to 'PushSettings'.")

        # Gaurd against a forgotten call to "Pop".
        if cls.storedFormatSettings is not None:
            cls.PopSettings()

        cls.storedFormatSettings = cls.formatSettings
        cls.formatSettings       = formatSettings


    @classmethod
    def PopSettings(cls):
        """
        Restores the previous settings.

        Returns
        -------
        None.
        """
        if cls.storedFormatSettings is None:
            raise Exception("Invalid call to PopSettings.  Settings must first be pushed before popping.")

        cls.formatSettings       = cls.storedFormatSettings
        cls.storedFormatSettings = None


    @classmethod
    def GetListOfPlotStyles(self) -> list:
        """
        Get a list of the plot styles.

        Returns
        -------
        styles : list
            A list of plot styles.
        """
        directory  = Path.GetDirectory(__file__)
        styleFiles = Path.GetAllFilesByExtension(directory, "mplstyle")
        styles     = [os.path.splitext(styleFile)[0] for styleFile in styleFiles]
        return styles


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
        return plt.rcParams["font.size"]


    @classmethod
    def GetScaledAnnotationSize(cls) -> float:
        """
        Gets the annotation font size adjusted with the scaling factor.

        Parameters
        ----------
        None.

        Returns
        -------
        : float
            Scaled annotation size.
        """
        return cls.formatSettings.Scale*cls.formatSettings.AnnotationSize


    @classmethod
    def Format(cls):
        """
        Sets the font sizes, weights, and other properties of a plot.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # If the file does not contain a directory, assume the same directory as this file.
        # If the file does not contain a file extension, assume a default.
        parameterFile = cls.formatSettings.ParameterFile

        # If the parameter file is one of the built in ones, we don't have to do anything.
        if not parameterFile in plt.style.available:
            # If the full path is specificied, we are ok.
            # If the full path is not specified, it is assumed the file was shipped with this library and we need to locate it.
            if not Path.ContainsDirectory(parameterFile):
                parameterFile = os.path.join(Path.GetDirectory(__file__), parameterFile)

            if not parameterFile.endswith(".mplstyle"):
                parameterFile += ".mplstyle"

        # Reset so we start from a clean slate.  This prevent values that were changed previously from unexpectedly leaking
        # through to another plot.  This resets everything then applies new base formatting (matplotlib, seaborn, et cetera).
        cls.ResetMatPlotLib()
        cls.currentColor = -1

        # Establish the parameters specified in the input file.
        plt.style.use(parameterFile)

        # Apply override, if they exist.
        if cls.formatSettings.Overrides is not None:
            plt.rcParams.update(cls.formatSettings.Overrides)

        # Apply scaling.
        parameters = {
            "font.size"              : cls._ScaleFontSize(plt.rcParams["font.size"]),
            "figure.titlesize"       : cls._ScaleFontSize(plt.rcParams["figure.titlesize"]),
            "legend.fontsize"        : cls._ScaleFontSize(plt.rcParams["legend.fontsize"]),
            "legend.title_fontsize"  : cls._ScaleFontSize(plt.rcParams["legend.title_fontsize"]),
            "axes.titlesize"         : cls._ScaleFontSize(plt.rcParams["axes.titlesize"]),
            "axes.labelsize"         : cls._ScaleFontSize(plt.rcParams["axes.labelsize"]),
            "xtick.labelsize"        : cls._ScaleFontSize(plt.rcParams["xtick.labelsize"]),
            "ytick.labelsize"        : cls._ScaleFontSize(plt.rcParams["ytick.labelsize"]),
            "axes.linewidth"         : plt.rcParams["axes.linewidth"]*cls.formatSettings.Scale,                   # Axis border.
            "patch.linewidth"        : plt.rcParams["patch.linewidth"]*cls.formatSettings.Scale,                  # Legend border.
            "lines.linewidth"        : plt.rcParams["lines.linewidth"]*cls.formatSettings.Scale,
            "lines.markersize"       : plt.rcParams["lines.markersize"]*cls.formatSettings.Scale,
            "axes.labelpad"          : plt.rcParams["axes.labelpad"]*cls.formatSettings.Scale,
        }
        plt.rcParams.update(parameters)


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
        plt.rcdefaults()


    @classmethod
    def _ScaleFontSize(cls, size) -> float:
        """
        Scale a font by the scale.  Checks for missing values and converts values that are strings to their numerical values.

        Parameters
        ----------
        size : None, string, or float
            The size of the font.  If None is supplied, the default value is used.

        Returns
        -------
        : float
            The font size converted to a numerical value and scaled.
        """
        if size is None:
            size = matplotlib.font_manager.fontManager.get_default_size();

        if type(size) is str:
            size = cls.ConvertFontRelativeSizeToPoints(size)

        return size*cls.formatSettings.Scale


    @classmethod
    def ConvertFontRelativeSizeToPoints(cls, relativeSize) -> float:
        """
        Converts a relative size (large, small, medium, et cetera) to a numerical value.

        Parameters
        ----------
        relativeSize : string
            A Matplotlib relative font size string.

        Returns
        -------
        : float
            The font size as a flaot.
        """
        if type(relativeSize) is not str:
            raise Exception("The relative font size must be a string.")

        defaultSize = matplotlib.font_manager.fontManager.get_default_size();
        scalings    = matplotlib.font_manager.font_scalings

        if not relativeSize in scalings:
            raise Exception("Not a valid relative font size.")

        return scalings[relativeSize] * defaultSize


    @classmethod
    def NewTopAndBottomAxisFigure(cls, title:str, topFraction:float=0.25) -> tuple[fig.Figure, tuple[ax.Axes, ax.Axes]]:
        """
        Creates a new figure that has two axes, one above another.

        Parameters
        ----------
        title : str
            Figure title.
        topFraction : float
            The fraction of the total space the top figure should use.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        (boxAxis, historgramAxis) : matplotlib.axes.Axes tuple
            The top axis and bottom axis, respectively, for the box plot and histogram.
        """
        # Check input.
        if topFraction <= 0 or topFraction >= 1.0:
            raise Exception("Top percentage out of range.")

        # The format setup needs to be run first.
        cls.Format()

        figure, (boxAxis, histogramAxis) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios" : (topFraction, 1-topFraction)})

        figure.suptitle(title)

        return figure, (boxAxis, histogramAxis)


    @classmethod
    def NewSideBySideAxisFigure(cls, title:str, width:float=15, height:float=5) -> tuple[fig.Figure, tuple[ax.Axes, ax.Axes]]:
        """
        Creates a new figure that has two axes, one above another.

        Parameters
        ----------
        title : str
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
        cls.Format()

        figure, (leftAxis, rightAxis) = plt.subplots(1, 2)

        figure.set_figwidth(width)
        figure.set_figheight(height)


        figure.suptitle(title)

        return figure, (leftAxis, rightAxis)


    @classmethod
    def NewMultiXAxesFigure(cls, numberOfAxes:int) -> tuple[fig.Figure, list]:
        """
        Creates a new figure that has multiple axes that are on top of each other.  The axes have an aligned (shared) y-axis.

        Parameters
        ----------
        numberOfAxes : int
            The number of axes to create.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        (axis1, axis2, ..., axisN) : maptplotlib.axes.Axes list
            The axes.
        """
        # The format setup needs to be run first.
        cls.Format()

        figure = plt.figure()
        axeses = [figure.gca()]

        for i in range(1, numberOfAxes):
            axeses.append(axeses[0].twiny())

            # Ideally, we would calculate an offset based on all the text sizes and spacing, but that seems challenging.
            # axeses[i].xaxis.label.get_size()
            # plt.rcParams["axes.titlesize"]  plt.rcParams["axes.labelsize"] plt.rcParams["xtick.labelsize"]
            # Instead, we will use a linear scaling with a y-intercept that doesn't pass through zero.  This seems to work reasonable well.
            s1     = 55                     # First point selected at a plot scale of 1.0.  This is the size in points.
            s2     = 25                     # Second point selected at a plot scale of 0.25.  This is the size in points.
            m      = 4/3.0*(s1-s2)          # Slope.
            y0     = (4.0*s2-s1) / 3.0      # Y-intercept.
            offset = m * cls.formatSettings.Scale + y0
            axeses[i].spines["top"].set_position(("outward", offset))

        # Move the first axis ticks and label to the top.
        axeses[0].xaxis.tick_top()
        axeses[0].xaxis.set_label_position("top")

        return figure, axeses


    @classmethod
    def NewMultiYAxesFigure(cls, numberOfAxes:int) -> tuple[fig.Figure, list]:
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
        cls.Format()

        figure = plt.figure()
        axeses = [figure.gca()]

        for i in range(1, numberOfAxes):
            # Create the remaining axis and specify that the same x-axis should be used.
            axeses.append(axeses[0].twinx())

            # Off set the new y-axis labels to the right of the previous.
            offset = 1.0 + (i-1)*0.12
            axeses[i].spines["right"].set_position(("axes", offset))

        # Change the drawing order of axes so the first one created is on top.
        AxesHelper.SetZOrderOfMultipleAxesFigure(axeses)

        return figure, axeses


    @classmethod
    def ConvertKeyWordArgumentsToSeriesSets(cls, numberOfSets:int, **kwargs) -> list:
        """
        Converts key word arguments into a set of key word arguments.

        Example:
            ConvertKeyWordArgumentsToSeriesSets(2, color="r")
            Output:
                [{color:"r"}, {color:"r"}]

                ConvertKeyWordArgumentsToSeriesSets(2, color=["r", "g"], linewidth=3)
                Output:
                    [{color:"r", linewidth=3}, {color:"g", linewidth=3}]

        Parameters
        ----------
        numberOfSets : int
            The number of output key word argument sets.
        **kwargs : keyword arguments
            The key word arguments to convert.

        Returns
        -------
        keyWordArgumentSets : list
            A list of length numberOfSets that contains individual key word argument dictionaries.
        """
        keyWordArgumentSets = []

        for i in range(numberOfSets):
            seriesKwargs = {}

            for key, value in kwargs.items():
                match value:
                    case int() | float() | str() | None:
                        seriesKwargs[key] = value

                    case list():
                        seriesKwargs[key] = value[i]

                    case _:
                        raise Exception("Unknown type found.\nType:  "+str(type(value))+"\nKey:   "+str(key)+"\nValue: "+str(value)+"\nSupplied kwargs:\n"+str(kwargs))

            keyWordArgumentSets.append(seriesKwargs)
        return keyWordArgumentSets


    @classmethod
    def NewArtisticFigure(cls, parameterFile:str=None) -> tuple[fig.Figure, ax.Axes]:
        """
        Create a new artistic plot.

        Parameters
        ----------
        parameterFile : str, optional
            A Matplotlib parameter style file. The default is None.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        axes : matplotlib.axes.Axes
            The axes of the plot.
        """
        if parameterFile is None:
            parameterFile = "artistic"

        cls.PushSettings(parameterFile=parameterFile)
        cls.Format()

        figure  = plt.gcf()
        axes    = plt.gca()

        # Zero lines.
        axes.axhline(y=0, color="black", linewidth=3.6*cls.formatSettings.Scale)
        axes.axvline(x=0, color="black", linewidth=3.6*cls.formatSettings.Scale)
        AxesHelper.AddArrows(axes, color="black")

        # Erase axis numbers (labels).
        axes.set(xticks=[], yticks=[])

        cls.PopSettings()

        return figure, axes


    @classmethod
    def GetColorCycle(cls, lineColorCycle:str=None, numberFormat:str="RGB") -> list:
        """
        Gets the default Matplotlib colors in the color cycle.

        Parameters
        ----------
        lineColorCycle : str, optional
            The color cycle to use for line colors.
                None      : The color cycle is taken from the format settings.
                "pyplot"  : The Matplotlib default color cycle is returned.
                "seaborn" : The default Seaborn color cycle is returned.
            The default is None.
        numberFormat : str, optional
            The number format to return the colors as.  The options are "RGB" or "hex".  The
            default is "RGB".

        Returns
        -------
        : list
            Colors in the color cycle.
        """
        numberFormat = numberFormat.lower()
        if numberFormat != "rgb" and numberFormat != "hex":
            raise Exception("The number format specified is not valid.\nRequested format: "+numberFormat)

        if lineColorCycle is None:
            lineColorCycle = cls.formatSettings.LineColorCycle

        if lineColorCycle == "pyplot":
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors     = prop_cycle.by_key()['color']

            if numberFormat == "rgb":
                colors = cls.ListOfHexToRgb(colors)

        elif lineColorCycle == "seaborn":
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

            if numberFormat == "hex":
                colors = cls.ListOfRgbToHex(colors)

        else:
            raise Exception("Unkown color style requested.\nRequested style: "+lineColorCycle)

        return colors


    @classmethod
    def ListOfHexToRgb(cls, colors:list|tuple) -> list:
        """
        Convert a list of colors represented as hexadecimal strings into RGB colors.

        Parameters
        ----------
        colors : array like of strings
            An array like series of strings that are hexadecimal values representing colors.

        Returns
        -------
        : List of tuples.
            RGB colors in a List of colors in a tuple.
        """
        return [ImageColor.getrgb(color) for color in colors]


    @classmethod
    def RgbToHex(cls, color:list|tuple) -> str:
        """
        Converts an RGB color to a hexadecimal string color.

        Parameters
        ----------
        color : array like
            A RGB color.

        Returns
        -------
        : str
            A hexadecimal color.
        """
        if isinstance(color[0], float):
            color = [math.floor(255*x) for x in color]

        return "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])


    @classmethod
    def ListOfRgbToHex(cls, colors:list) -> list:
        """
        Converts an list of RGB colors to a list of hexadecimal string colors.

        Parameters
        ----------
        colors : array like of array like
            A list of RGB colors.

        Returns
        -------
        : list of string
            List of hexadecimal colors.
        """
        if isinstance(colors[0][0], float):
            colors = [[math.floor(255*x) for x in color] for color in colors]

        return ["#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2]) for color in colors]


    @classmethod
    def NextColor(cls):
        cls.currentColor += 1
        return cls.GetColorCycle()[cls.currentColor]


    @classmethod
    def NextColorAsHex(cls):
        return cls.RgbToHex(cls.NextColor())


    @classmethod
    def CurrentColor(cls):
        return cls.GetColorCycle()[cls.currentColor]


    @classmethod
    def CurrentColorAsHex(cls):
        return cls.RgbToHex(cls.CurrentColor())


    @classmethod
    def GetColor(cls, color:int) -> tuple:
        return cls.GetColorCycle()[color]


    @classmethod
    def CategoryTitle(cls, categoryName:str) -> str:
        """
        Formats a string as a category title.  It is converted to title case, quotes
        added around the category name and "Category" added as a suffix.

        Parameters
        ----------
        categoryName : str
            Category name to convert to a title.

        Returns
        -------
        title : str
            The category converted to a title.
        """
        return "\"" + categoryName.title() + "\"" + " Category"


    @classmethod
    def GetDefaultOutputDirectory(cls) -> str:
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
    def SavePlot(cls, saveFileName:str, figure:matplotlib.figure.Figure=None, transparent=False):
        """
        Saves a plot with a set of default parameters.

        Parameters
        ----------
        saveFileName : str
            The (optionally) path and file name to save the image to.
        figure : matplotlib.figure.Figure, optional
            The figure to save.  If None is specified, the current figure will be used.  The default is None.
        transparent : bool, optional
            Specificies if the background of the plot should be transparent.  If True, the background will be set to transparent, if False, nothing no
            action is taken.  The default is False.

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
        if cls.usedefaultOutputDirectory:

            # Directory needs to exist.
            if not os.path.isdir(cls.defaultOutputDirectory):
                os.mkdir(cls.defaultOutputDirectory)

            # Update path.
            path = os.path.join(cls.defaultOutputDirectory, saveFileName)

        # And, finally, get down to the work.
        figure.savefig(path, dpi=500, transparent=transparent, bbox_inches="tight")


    @classmethod
    def SavePlotToBuffer(cls, figure=None, format="png", autoCrop=False, borderSize=0) -> BytesIO:
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
        plot : BytesIO
            Buffer with the figure written to it.
        """
        if figure == None:
            figure = plt.gcf()

        buffer = cls.SaveToBuffer(figure, "PNG" if autoCrop else format)

        if autoCrop:
            figure = Image.open(buffer).convert("RGB")
            buffer.close()
            figure = cls.CropWhiteSpace(figure, borderSize)
            buffer = cls.SaveToBuffer(figure, format)

        image     = buffer.getvalue()
        plot      = base64.b64encode(image)
        plot      = plot.decode("utf-8")

        buffer.close()

        return plot


    @classmethod
    def SaveToBuffer(cls, figure, format="PNG") -> BytesIO:
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
        buffer : BytesIO
            Buffer with the figure written to it.
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
            An image saved in a buffer.
        borderSize : int
            The size of the border, in pixels, to leave remaing around the edge.

        Returns
        -------
        image : BytesIO
            Buffer with the cropped image.
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