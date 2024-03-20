"""
Created on Septemper 28, 2023
@author: lance.endres
"""
from   copy                                                          import deepcopy

# Allow the return of this class as a type hint.  See: https://peps.python.org/pep-0673/
from   typing                                                        import Self


class FormatSettings():
    """
    A class that contains the settings used to format a new plot.
        It does contain the setting values.
        It does not do the formatting.

    See also PlotHelper.
    """


    def __init__(self, parameterFile:str="default", overrides:dict={}, scale:int=1.0, annotationSize:int=12, lineColorCycle:str="seaborn"):
        """
        Constructors

        Parameters
        ----------
        parameterFile : str, optional
            The path to a parameters (.mplstyle) file.  A parameter file is a file that contains MatPlotLib style settings.  The default is "default".
        overrides : dict, optional
            Overrides applied on top of those read from the parameters file. The default is {}.
        scale : int, optional
            A scale applied to certain settings like line weights and fonts sizes.  It is used to applied minor adjustments to the settings from a parameter
            file. The default is 1.0.
        annotationSize : int, optional
            The font size (in points) used for labeling (annotating) a figure.  Annotations are additional information that is applied to a plot.  The default is 15.
        lineColorCycle : str, optional
            Name of the color cycle to use. The default is "seaborn".

        Returns
        -------
        None.
        """
        # Overrides needs to exist before calling update.
        self.overrides = {}
        self.Update(parameterFile, deepcopy(overrides), scale, annotationSize, lineColorCycle)


    def Copy(self) -> Self:
        """
        Copy.  Creates a new instance with values copied from this instance.

        Returns
        -------
        : FormatSettings
            A new FormatSettings object.
        """
        return FormatSettings(self.parameterFile, self.overrides, self.scale, self.annotationSize, self.lineColorCycle)


    def Update(self, parameterFile:str=None, overrides:dict=None, scale:int=None, annotationSize:int=None, lineColorCycle:str=None) -> Self:
        """
        Update the settings with new ones.

        Parameters
        ----------
        parameterFile : str, optional
            The path to a parameters (.mplstyle) file.  A parameter file is a file that contains MatPlotLib style settings.  The default is "default".
        overrides : dict, optional
            Overrides applied on top of those read from the parameters file. The default is {}.
        scale : int, optional
            A scale applied to certain settings like line weights and fonts sizes.  It is used to applied minor adjustments to the settings from a parameter
            file. The default is 1.0.
        annotationSize : int, optional
            The font size (in points) used for labeling (annotating) a figure.  Annotations are additional information that is applied to a plot.  The default is 15.
        lineColorCycle : str, optional
            Name of the color cycle to use. The default is "seaborn".

        Returns
        -------
        self : FormatSettings
            This instance.
        """
        # Parameter file.
        if parameterFile is not None:
            self.parameterFile               = parameterFile

        # Overrides of rcParams in the parameter file.
        if overrides is not None:
            self.overrides.update(overrides)

        # Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot. The default is 1.0.
        if scale is not None:
            self.scale                       = scale

        # Alternate font size used for annotations, labeling, et cetera.
        if annotationSize is not None:
            self.annotationSize              = annotationSize

        # Format style.  This is the default, it can be overridden in the call to "Format".
        if lineColorCycle is not None:
            self.lineColorCycle              = lineColorCycle

        return self


    @property
    def ParameterFile(self):
        return self.parameterFile


    @ParameterFile.setter
    def ParameterFile(self, parameterFile):
        self.parameterFile = parameterFile


    @property
    def Overrides(self):
        return self.overrides


    @Overrides.setter
    def Overrides(self, overrides):
        self.overrides = overrides


    @property
    def Scale(self):
        return self.scale


    @Scale.setter
    def Scale(self, scale):
        self.scale = scale


    @property
    def AnnotationSize(self):
        return self.annotationSize


    @AnnotationSize.setter
    def AnnotationSize(self, annotationSize):
        self.annotationSize = annotationSize


    @property
    def LineColorCycle(self):
        return self.lineColorCycle


    @LineColorCycle.setter
    def LineColorCycle(self, lineColorCycle):
        self.lineColorCycle = lineColorCycle