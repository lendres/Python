"""
Created on Aug 27, 2022
@author: Lance A. Endres
"""
import numpy                                     as np
from   adjustText                                import adjust_text

class AnnotationHelper():
    # https://adjusttext.readthedocs.io/en/latest/#
    # https://stackoverflow.com/questions/67289586/django-overriding-display-names-for-foreign-key-field-in-modelform


    def __init__(self, parameters=None):
        # List of annotation that were created.  Saved so adjustments can be made.
        self.annotations     = []

        # Typically, you will be annotating most or all labels the same way.  Therefore, instance level defaults are
        # provided.  Defaults can be overriden during function calls if required.
        # Default annotation foramt.
        self.defaults = {
            "annotationFormat"  : {},
            "formatString"      : "{0:0.0f}",
            "positionOffset"    : [0, 0]
        }

        self.defaults = self._GetProperties(parameters)


    def AddMaxAnnotation(self, axis, lines, overrides=None):
        if type(lines) is not list:
            lines = [lines]
            
        for line in lines:
            self.AddAnnotation(axis, line, max, overrides)


    def AddAnnotation(self, axis, line, function, overrides):
        arguments = self._GetProperties(overrides)
        
        # Find the maximum value of the dependent variable, then find the assocated x (independent) value.
        y        = line.get_ydata()
        yValue   = function(y)

        # The numpy.where function will not perform as expected on lists of floats, so ensure everything
        # is an numpy.array to allow the comparison to perform correctly.  The returned value is a
        # tuple of array and type so "index[0][0]" extracts the first element out of the array in the tuple.
        index   = np.where(np.asarray(y) == np.asarray(yValue))
        index   = index[0][0]
        xValue  = line.get_xdata()
        xValue  = xValue[index]

        offsetPosition = arguments["positionOffset"]
        textPosition   = [xValue+offsetPosition[0], yValue+offsetPosition[1]]
        text           = arguments["formatString"].format(yValue)
        annotation     = axis.annotate(text, textPosition, **arguments["annotationFormat"])

        self.annotations.append(annotation)


    def AdjustAnnotations(self, **kwargs):
        adjust_text(self.annotations, **kwargs)


    def _GetProperties(self, overrides):
        if overrides is not None:
            values = self.defaults.copy()
            values.update(overrides)
            return values
        else:
            return self.defaults