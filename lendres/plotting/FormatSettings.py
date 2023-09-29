"""
Created on Septemper 28, 2023
@author: lance.endres
"""

class FormatSettings():


    def __init__(self, parameterFile:str="default", overrides:dict=None, scale:int=1.0, annotationSize:int=15, lineColorCycle:str="seaborn"):
        # Parameter file.
        self.parameterFile               = parameterFile

        # Overrides of rcParameters in the parameter file.
        self.overrides                   = overrides

        # Scaling parameter used to adjust the plot fonts, lineweights, et cetera for the output scale of the plot. The default is 1.0.
        self.scale                       = scale

        # Alternate font size used for annotations, labeling, et cetera.
        self.annotationSize              = annotationSize

        # Format style.  This is the default, it can be overridden in the call to "Format".
        self.lineColorCycle              = lineColorCycle


    def Copy(self):
        return FormatSettings(self.parameterFile, self.overrides, self.scale, self.annotationSize, self.lineColorCycle)


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