"""
Created on August 27, 2022
@author: Lance A. Endres
"""
import numpy                                                         as np
from   matplotlib.lines                                              import Line2D
from   adjustText                                                    import adjust_text
from   collections.abc                                               import Callable

from   lendres.plotting.PlotHelper                                   import PlotHelper
from   lendres.signalprocessing.SignalProcessing                     import SignalProcessing


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


    def __init__(self, formatString:str="{x:0.0f}, {y:0.0f}", **kwargs):
        """
        Contructor.

        Parameters
        ----------
        formatString : str, optional
            The format string used to generate the annotations. The default is "{x:0.0f}, {y:0.0f}".
        **kwargs : keyword arguments
            Keyword arguments passed to the annotation function.  See https://adjusttext.readthedocs.io/en/latest/

        Returns
        -------
        None.
        """
        # List of annotation that were created.  Saved so adjustments can be made.
        self.annotations          = []

        # Default annotation foramt string.
        self.formatString         = formatString

        self.adjustText           = False
        self.adjustTextKwargs     = {}

        # Typically, you will be annotating most or all labels the same way.  Therefore, instance level defaults are
        # provided.  Defaults can be overriden during function calls if required.
        # Default annotation format.
        self.defaults = {
            "size"                : PlotHelper.GetScaledAnnotationSize(),
            "fontweight"          : "bold",
            "xytext"              : (0, 2),
            "textcoords"          : "offset points",
            "horizontalalignment" : "center",
            "verticalalignment"   : "bottom"
        }

        self.defaults = self._CombineSettingsAndOverrides(kwargs)


    @property
    def FormatString(self):
        return self.formatString


    @FormatString.setter
    def FormatString(self, formatString):
        self.formatString = formatString


    def SetAdjustText(self, adjustText:bool=False, **kwargs):
        """
        Sets the values for adjusting the text to avoid other entities.

        Parameters
        ----------
        adjustText : bool, optional
            If True, an algorithm will adjust the annotations so they do not overlap other text or lines. The default is False.
        **kwargs : keyword arguments
            Keyword arguments passed to the text adjusting function.

        Returns
        -------
        None.
        """
        # You cannot use "xytext" (offsets) with "adjust_text" so we have to turn them off.
        if "xytext" in self.defaults:
            # ConsoleHelper().PrintWarning("The 'adjust_text' function is not compatible with the 'xytext' parameter of the annotation function.")
            self.defaults = self._CombineSettingsAndOverrides({"xytext" : None, "textcoords" : None})
        self.adjustText       = adjustText
        self.adjustTextKwargs = kwargs


    def AddMaximumAnnotation(self, lines:Line2D|list[Line2D]) -> tuple[list, list]:
        """
        Adds an annotation to the line(s) at the maximum Y value.

        Parameters
        ----------
        lines : Line2D or list of Line2D
            Line(s) returned by plotting on an axes.  The maximum value of each line is found and annotated.
        number : int, optional
            Specifies how many of the top values should be labeled.  The default is 1.

        Returns
        -------
        indices : list
            The indices of the labeled values.
        yValues : list
            The values of the labeled values.
        """
        return self.AddAnnotationsByFunction(lines, self._GetMax)


    def _GetMax(self, y, **kwargs) -> tuple[list, list]:
        # If there is a "nan" in the data, it treats that as the max and returns nothing for the index.
        # Therefore, replace the "nan"s with zeros.
        y       = np.nan_to_num(y)

        yMax    = max(y)
        index   = np.where(np.asarray(y) == np.asarray(yMax))
        index   = index[0][0]
        return [index], [yMax]


    def AddAnnotationToLastValue(self, lines:Line2D|list) -> tuple[list, list]:
        """
        Labels the last value of the line.

        Parameters
        ----------
        lines : Line2D or list of Line2D
            Line(s) returned by plotting on an axes.  The maximum value of each line is found and annotated.

        Returns
        -------
        indices : list
            The indices of the labeled values.
        yValues : list
            The values of the labeled values.
        """
        return self.AddAnnotationsByFunction(lines, self._GetLastValue)


    def _GetLastValue(self, y, **kwargs) -> tuple[list, list]:
        index   = len(y) -1
        return [index], [y[index]]


    def AddPeakAnnotations(self, lines:Line2D|list, sortBy:str="localheight", **kwargs) -> tuple[list, list]:
        """
        Labels the local maximum values.

        Parameters
        ----------
        lines : Line2D or list of Line2D
            Line(s) returned by plotting on an axes.  The maximum value of each line is found and annotated.
        sortBy : str, optional
            The method used to sort the peaks.
                localheight  : The peaks are sorted according to how high are they above the local terrain.  In this method
                    the highest peak (locally) is not necessarily the highest peak overall (globally).
                globalheight : The peaks are sorted according to which ones are highest overall (globally).
            The default is "localheight".
        **kwargs : keyword arguments
            Keyword arguments passed to the SignalProcessing.GetPeaks function.

        Returns
        -------
        indices : list
            The indices of the labeled values.
        yValues : list
            The values of the labeled values.
        """
        return self.AddAnnotationsByFunction(lines, SignalProcessing.GetPeaks, sortBy=sortBy, **kwargs)


    def AddAnnotationsByFunction(self, lines:Line2D|list[Line2D], function:Callable, **kwargs) -> tuple[list, list]:
        """
        Applies the function to the line to find the value (Y) and location (X) of a point on the line.  That point
        is then annotated on the specified axes using a combination of the default and overrides settings.

        Parameters
        ----------
        lines : Line2D or list of Line2D
            Line(s) returned by plotting on an axes.
        function : function
            A function that can be applied to an array like object.  The function finds a specific Y value to annotate.
            The function must return a list, tuple, or other iterable object.

        Returns
        -------
        indices : list
            The indices of the labeled values.
        yValues : list
            The values of the labeled values.
        """
        if type(lines) is not list:
            lines = [lines]

        for line in lines:
            # Find the maximum value of the dependent variable, then find the assocated x (independent) value.
            y                 = line.get_ydata()
            indices, yValues  = function(y, **kwargs)

            for index, yValue in zip(indices, yValues):
                # The numpy.where function will not perform as expected on lists of floats, so ensure everything
                # is an numpy.array to allow the comparison to perform correctly.  The returned value is a
                # tuple of array and type so "index[0][0]" extracts the first element out of the array in the tuple.
                xValue  = line.get_xdata()
                xValue  = xValue[index]

                textPosition   = [xValue, yValue]
                text           = self.formatString.format(x=xValue, y=yValue)
                annotation     = line.axes.annotate(text, textPosition, **self.defaults)

                self.annotations.append(annotation)

        if self.adjustText:
            self._AdjustAnnotations()

        return indices, yValues


    def _AdjustAnnotations(self):
        """
        Takes all the annotations that have been created and passes them to the "adjustText" library for position refinement.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        adjust_text(self.annotations, **self.adjustTextKwargs)


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