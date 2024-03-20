"""
Created on August 27, 2022
@author: Lance A. Endres
"""
import numpy                                                         as np
from   adjustText                                                    import adjust_text
from   scipy.signal                                                  import find_peaks
import heapq

from   lendres.plotting.PlotHelper                                   import PlotHelper
from   lendres.io.ConsoleHelper                                      import ConsoleHelper

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


    def AddMaximumAnnotation(self, lines):
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
        None.
        """
        self.AddAnnotationsByFunction(lines, self._GetMax)


    def _GetMax(self, y, **kwargs):
        yMax    = max(y)
        index   = np.where(np.asarray(y) == np.asarray(yMax))
        index   = index[0][0]
        return [index], [yMax]


    def AddPeakAnnotations(self, lines, sortBy="localheight", **kwargs):
        self.AddAnnotationsByFunction(lines, self._GetPeaks, sortBy=sortBy, **kwargs)


    def _GetPeaks(self, y, number=6, sortBy="localheight", **kwargs):
        # Create default arguments, then override/update with any specified arguments.
        arguments = {"distance" : 4, "prominence" : 0.1}
        arguments.update(kwargs)

        # We find the peaks.
        # The distance argument is provided to group values that are extremely close together.  I.e., a shallow slow with small local peaks is not of interest.
        # The height argument is provided only to get the algorithm to return the relative peak prominences.  The relative heights/prominences are used as an 'importance' factor in sorting.
        # The indices of the peaks are the first firsted value from find_peaks.
        peakResults         = find_peaks(y, **arguments)

        # Extract the top values from the results.  The prominances are the local heights and the first entry returned in peakResults are the y values.
        # The output of find_peaks is [[y_values], dict{}]
        localHeights        = (peakResults[1])["prominences"]
        peakIndices         = peakResults[0]

        # Sort the peakIndices (absolute heights) according to their local height value (how high are they above the surrounding local territory.
        # The top values are defined as those with the largest local peak height.
        match sortBy:
            case "localheight":
                largestPeaks = heapq.nlargest(number, zip(localHeights, peakIndices))
            case "globalheight":
                largestPeaks = heapq.nlargest(number, zip(y[peakIndices], peakIndices))
            case _:
                raise Exception("The 'sortBy' parameter is not valid.")

        # Extract the y (absolute heights) from the sorted results.
        largestPeaksIndices = [peakCouple[1] for peakCouple in largestPeaks]
        largestYValues      = y[largestPeaksIndices]

        return largestPeaksIndices, largestYValues


    def AddAnnotationsByFunction(self, lines, function, **kwargs):
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
        None.
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