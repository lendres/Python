"""
Created on Aug 27, 2022
@author: Lance A. Endres
"""
import numpy                                     as np
from   adjustText                                import adjust_text

class AnnotationHelper():
    """
    This is a helper class for annotating plots.  Its functionality includes:
        - Provide common settings for repeat annotations.
        - Add functionality like the ability to adjust the position of the text.  This can be used to move it slightly
          away from a line, for example.
        - Makes use the adjustText library to prevent overlap of text.
          See: # https://adjusttext.readthedocs.io/en/latest/#

    This class assumes that many annotations of the same font and string format are used.  Therefore, there are default, class level,
    setttings to control those options (every annotation gets created the same way).  However, those can be overridden when creating
    an annotation for individual control.

    For information about annotation kwargs, see:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html
    """


    def __init__(self, settings=None):
        """
        Contructor.

        Parameters
        ----------
        settings : dictionary
            Settings used in the creation of the annotation.  They include:
                annotationKwargs : Arguments passed to matplotlib.Axes.annotate.
                formatString     : Format applied to text.  Allows things like rounding numbers and adding prefixes or
                                   suffixes.  For example, "Max {0:0.0f}" turns 36.87 into "Max 37".
                positionOffset   : Used to move the text aways from lines so it doesn't touch.  If using the "AdjustAnnotations"
                                   option it might be better to not use this settings.  Experiments might be required to find the
                                   best result.

        Returns
        -------
        None.
        """
        # List of annotation that were created.  Saved so adjustments can be made.
        self.annotations     = []

        # Typically, you will be annotating most or all labels the same way.  Therefore, instance level defaults are
        # provided.  Defaults can be overriden during function calls if required.
        # Default annotation foramt.
        self.defaults = {
            "annotationKwargs"  : {},
            "formatString"      : "{0:0.0f}",
            "positionOffset"    : [0, 0]
        }

        self.defaults = self._CombineSettingsAndOverrides(settings)


    def AddMaxAnnotation(self, lines, overrides=None):
        """
        Adds an annotation to the line(s) at the maximum Y value.

        Parameters
        ----------
        lines : Line2D or list of Line2D
            Line(s) returned by plotting on an axes.  The maximum value of each line is found and annotated.
        overrides : dictionary, optional
           Any of the values specified by the "settings" argument in the constructor can be present.  They are used
           to override the default settings.

        Returns
        -------
        None.

        """
        if type(lines) is not list:
            lines = [lines]

        for line in lines:
            self.AddAnnotationByFunction(line, max, overrides)


    def AddAnnotationByFunction(self, line, function, overrides=None):
        """
        Applies the function to the line to find the value (Y) and location (X) of a point on the line.  That point
        is then annotated on the specified axes using a combination of the default and overrides settings.

        Parameters
        ----------
        line : Line2D
            DESCRIPTION.
        function : function
            A function that can be applied to an array like object.  The function finds a specific Y value to annotate.
        overrides : dictionary, optional
           Any of the values specified by the "settings" argument in the constructor can be present.  They are used
           to override the default settings.

        Returns
        -------
        None.

        """
        arguments = self._CombineSettingsAndOverrides(overrides)

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
        annotation     = line.axes.annotate(text, textPosition, **arguments["annotationKwargs"])

        self.annotations.append(annotation)


    def AdjustAnnotations(self, **kwargs):
        """
        Takes all the annotations that have been created and passes them to the "adjustText" library for position refinement.

         Parameters
        ----------
        kwargs : dictionary
            Arguments passed to the "adjustText" library.

        Returns
        -------
        None.
        """
        adjust_text(self.annotations, **kwargs)


    def _CombineSettingsAndOverrides(self, overrides):
        """
        Applies the overrides to the default settings and returns the result.  It overwrites the defaults with the overrides.

        Parameters
        ----------
        overrides : dictionary
            Any of the values specified by the "settings" argument in the constructor can be present.

        Returns
        -------
        dictionary
            The settings with any overrides applied.
        """
        if overrides is not None:
            values = self.defaults.copy()
            values.update(overrides)
            return values
        else:
            return self.defaults