"""
Created on Septemper 28, 2023
@author: lance.endres
"""
from copy                                                            import deepcopy


class FormatSettings():


    def __init__(self, parameterFile:str="default", overrides:dict={}, scale:int=1.0, annotationSize:int=15, lineColorCycle:str="seaborn"):
        self.Set(parameterFile, deepcopy(overrides), "overwrite", scale, annotationSize, lineColorCycle)


    def Copy(self):
        return FormatSettings(self.parameterFile, self.overrides, self.scale, self.annotationSize, self.lineColorCycle)


    def Set(self, parameterFile:str=None, overrides:dict=None, overrideUpdateMethod="overwrite", scale:int=None, annotationSize:int=None, lineColorCycle:str=None):
        # Parameter file.
        if parameterFile is not None:
            self.parameterFile               = parameterFile

        # Overrides of rcParameters in the parameter file.
        if overrides is not None:
            match overrideUpdateMethod:
                case "overwrite":
                    self.overrides           = overrides
                case "update":
                    self.overrides.update(overrides)
                case _:
                    raise Exception("Invalid 'overrideUpdateMethod' argument provided to 'FormatSettings'.")

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