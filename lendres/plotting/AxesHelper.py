"""
Created on December 4, 2021
@author: Lance A. Endres
"""
import numpy                                                         as np
import matplotlib.pyplot                                             as plt


class AxesHelper():


    @classmethod
    def Label(cls, axeses, title, xLabels, yLabels="", titleSuffix:str=None):
        """
        Add title, x-axis label, and y-axis label.  Allows for multiple axes to be labeled at once.

        Parameters
        ----------
        axeses : figure or matplotlib.axes.Axes or array like of figure/axes
            The object(s) to label.
        title : TYPE
            Main plot title.
        xLabels : string or array like of strings
            X-axis label(s).  If axeses is an array, xLabels can be an array of the same length.
        yLabels : str or array like of str, optional
            Y-axis label(s).  Default is a blank string.  If axeses is an array, ylabels can be an array of the same length.
        titleSuffix : str or None, optional
            If supplied, the string is appended as a second line to the title.  Default is none.

        Returns
        -------
        None.
        """
        # Create the title.
        if titleSuffix != None:
            title += "\n" + titleSuffix

        method = "single"

        if isinstance(axeses, list):
            # Check for both multiple x and y labels.
            if isinstance(xLabels, list) and isinstance(yLabels, list):
                if len(xLabels) != len(axeses) or len(yLabels) != len(axeses):
                    raise Exception("Invalid sizes of axeses, x labels, and y labels.")
                method = "multiple"

            # Check for multiple x labels.
            elif isinstance(xLabels, list):
                if len(xLabels) != len(axeses):
                    raise Exception("Invalid sizes of axeses and x labels.")
                method = "multipleX"

            # Check for multiple y labels.
            elif isinstance(yLabels, list):
                if len(yLabels) != len(axeses):
                    raise Exception("Invalid sizes of axeses and y labels.")
                method = "multipleY"

            # Unknown configuration.
            else:
                raise Exception("Invalid types of axeses, x labels, and y labels.")

        match method:
            case "single":
                axeses.set(title=title, xlabel=xLabels, ylabel=yLabels)
            case "multiple":
                for instance, xLabel, yLabel in zip(axeses, xLabels, yLabels):
                    instance.set(xlabel=xLabel, ylabel=yLabel)
            case "multipleX":
                axeses[0].set(title=title, ylabel=yLabels)
                for instance, xLabel in zip(axeses, xLabels):
                    instance.set(xlabel=xLabel)
            case "multipleY":
                axeses[0].set(title=title, xlabel=xLabels)
                for instance, yLabel in zip(axeses, yLabels):
                    instance.set(ylabel=yLabel)


    @classmethod
    def RotateXLabels(cls, xLabelRotation, axes=None):
        """
        Rotate the x-axis labels.

        Parameters
        ----------
        xLabelRotation : float
            Rotation of x labels.  If none is passed, nothing is done.
          axes : matplotlib.axes.Axes, optional
              The axes to change the x-axis label rotation.  If None, the current axes is used.

        Returns
        -------
        None.
        """
        savedCurrentAxes = plt.gca()

        if axes is not None:
            plt.sca(axes)

        # Option to rotate the x axis labels.
        if xLabelRotation is not None:
            plt.xticks(rotation=xLabelRotation, ha="right")

        plt.sca(savedCurrentAxes)


    @classmethod
    def SetZOrderOfMultipleAxesFigure(cls, axes):
        """
        Puts the left hand axes of a multiple y-axis plot on top of the z-order.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes.  The axes with a y-axis on the left is in axes[0].  The axes with the y-axis on
            the right are in axes[0] ... axes[N].

        Returns
        -------
        None.
        """
        # This is necessary to have the axes with the left y-axis show in front of the axes with the right y-axis.
        # In order to do this, two things are required:
        #    1) Reverse the z order so that the left axes is drawn above (after) the right axes.
        #    2) Reverse the patch (background) transparency.  The patch of the axes in front (left) has to be
        #       transparent.  We want the patch of the axes in back to be the same as before, so the alpha has
        #       to be taken from the left and set on the right.

        # It seems that the right axes can have an alpha of "None" and be transparent, but if we set that on
        # the left axes, it does not produce the same result.  Therefore, if it is "None", we default to
        # completely transparent.
        alphaSave  = axes[1].patch.get_alpha()
        alphaSave  = 0 if alphaSave is None else alphaSave

        # We use axes[-1] because it is the last axes on the right side and should be the highest in the order.  This
        # is an assumption.  The safer thing to do would be to loop through them all and retrieve the highest z-order.
        # Typically, they default to all the same, so this should be ok.
        maxZOrder = len(axes) - 1 + axes[-1].get_zorder()

        # Reverse the order.
        for i in range(0, len(axes)):
            axes[i].set_zorder(maxZOrder-i)
            axes[i].patch.set_alpha(axes[0].patch.get_alpha())

        # Make the front axes transparent (or whatever the last one was before).
        axes[0].patch.set_alpha(alphaSave)


    @classmethod
    def AlignXAxes(cls, axeses, numberOfTicks=None):
        """
        Align the ticks (grid lines) for the x-axis of multiple axes.  A new set of tick marks is computed
        as a linear interpretation of the existing range.  The number of tick marks is the
        same for all axes.  By setting them both to the same number of tick marks (same
        spacing between marks), the grid lines are aligned.

        Parameters
        ----------
        axes : list
            list of axes objects whose y-axis ticks are to be aligned.

        numberOfTicks : None or integer
            The number of ticks to use on the axes.  If None, the number of ticks on the
            first axis is used.

        Returns
        -------
        tickSets : list
            A list of new ticks for each axes in axeses.
        """
        cls.AlignAxes(axeses, "x", numberOfTicks)


    @classmethod
    def AlignYAxes(cls, axeses, numberOfTicks=None):
        """
        Align the ticks (grid lines) for the y-axis of multiple axes.  A new set of tick marks is computed
        as a linear interpretation of the existing range.  The number of tick marks is the
        same for all axes.  By setting them both to the same number of tick marks (same
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
            A list of new ticks for each axes in axeses.
        """
        cls.AlignAxes(axeses, "y", numberOfTicks)


    @classmethod
    def AlignAxes(cls, axeses, which, numberOfTicks=None):
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
            first axes is used.

        Returns
        -------
        tickSets : list
            A list of new ticks for each axes in axeses.
        """
        if which == "x":
            tickSets = [axes.get_xticks() for axes in axeses]
        elif which == "y":
            tickSets = [axes.get_yticks() for axes in axeses]
        else:
            raise Exception("Invalid direction specified in \"AlignAxes\"")

        # If the number of ticks was not specified, use the number of ticks on the first axes.
        if numberOfTicks is None:
            numberOfTicks = len(tickSets[0])

        numberOfIntervals = numberOfTicks - 1

        # The first axes is remains the same.  Those ticks should already be nicely spaced.
        tickSets[0] = np.linspace(tickSets[0][0], tickSets[0][-1], numberOfTicks, endpoint=True)

        #####
        # This method needs to be adjusted to account for different scale.  E.g. 0.2-0.8 versus 20-80.
        #####
        # Create a new set of tick marks that have the same number of ticks for each axes.
        # We have to scale the interval between tick marks.  We want them to be nice numbers (not something
        # like 72.2351).  To do this, we calculate a new interval by rounding up the existing spacing.  Rounding
        # up ensures no plotted data is cut off by scaling it down slightly.
        for i in range(1, len(tickSets)):
            span     = (tickSets[i][-1] - tickSets[i][0])
            interval = np.ceil(span / numberOfIntervals)
            tickSets[i] = np.linspace(tickSets[i][0], tickSets[i][0]+interval*numberOfIntervals, numberOfTicks, endpoint=True)

        # Set ticks for each axes.
        if which == "x":
            for axes, tickSet in zip(axeses, tickSets):
                axes.set(xticks=tickSet, xlim=(tickSet[0], tickSet[-1]))
        elif which == "y":
            for axes, tickSet in zip(axeses, tickSets):
                axes.set(yticks=tickSet, ylim=(tickSet[0], tickSet[-1]))

        return tickSets


    @classmethod
    def ReverseYAxisLimits(cls, axes):
        """
        Switches the upper and lower limits so that the highest value is on top.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            Axes to change the limits on.

        Returns
        -------
        None.
        """
        tickSet = axes.get_yticks()
        axes.set_ylim((tickSet[-1], tickSet[0]))


    @classmethod
    def SetXAxisNumberOfTicks(cls, axes, numberOfTicks):
        """
        Sets the x-axes limits.  Allows specifying the number of ticks to use.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            Axes to change the limits on.
        numberOfTicks : int, optional
            The number of ticks (labeled points) to show. The default is None.

        Returns
        -------
        None.
        """
        tickSet       = axes.get_xticks()
        cls.SetXAxisLimits(axes, [tickSet[0], tickSet[-1]], numberOfTicks)


    @classmethod
    def SetXAxisLimits(cls, axes, limits:list=None, lowerLimit:float=None, upperLimit:float=None, numberOfTicks:int|str="same"):
        """
        Sets the x-axes limits.  Allows specifying the number of ticks to use.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            Axes to change the limits on.
        limits : array like of two values, optional
            The lower and upper limits of the axis.  The default is None
        lowerLimit : float, optional
            The lower limit of the axis. The default is None.
        upperLimit : float, optional
            The upper limit of the axis. The default is None.
        numberOfTicks : int or string, optional
            The number of ticks (labeled points) to show.
            If "same", then the existing number of ticks is used.
            If "plusone", then the number of ticks is calculated as (upper_limits - lower_limit + 1).  The default is "same".

        Returns
        -------
        None.
        """
        limits  = cls._ProcessAxisLimitsArgument(limits, lowerLimit, upperLimit, cls.GetXBoundaries(axes))
        tickSet = cls._ProcessAxisTicksArgument(limits, numberOfTicks, axes.get_xticks())

        axes.set_xticks(tickSet)
        axes.set_xlim((tickSet[0], tickSet[-1]))


    @classmethod
    def SetYAxisLimits(cls, axes, limits:list=None, lowerLimit:float=None, upperLimit:float=None, numberOfTicks:int|str="same"):
        """
        Sets the y-axis limits.  Allows specifying the number of ticks to use.

        Specify either the limits as a list, or one or both of lowerLimit and upperLimit.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            Axes to change the limits on.
        limits : array like of two values, optional
            The lower and upper limits of the axis.  The default is None
        lowerLimit : float, optional
            The lower limit of the axis. The default is None.
        upperLimit : float, optional
            The upper limit of the axis. The default is None.
        numberOfTicks : int or string, optional
            The number of ticks (labeled points) to show.
            If "same", then the existing number of ticks is used.
            If "plusone", then the number of ticks is calculated as (upper_limits - lower_limit + 1).  The default is "same".

        Returns
        -------
        None.
        """
        limits  = cls._ProcessAxisLimitsArgument(limits, lowerLimit, upperLimit, cls.GetYBoundaries(axes))
        tickSet = cls._ProcessAxisTicksArgument(limits, numberOfTicks, axes.get_yticks())

        axes.set_yticks(tickSet)
        axes.set_ylim((tickSet[0], tickSet[-1]))


    @classmethod
    def _ProcessAxisLimitsArgument(cls, limits:list, lowerLimit:float, upperLimit:float, boundaries:list):
        if limits is None:
            limits = boundaries

            if lowerLimit is not None:
                limits[0] = lowerLimit

            if upperLimit is not None:
                limits[1] = upperLimit
        return limits


    @classmethod
    def _ProcessAxisTicksArgument(cls, limits:list, numberOfTicks:int|str, ticks:list):
        match numberOfTicks:
            case "same":
                numberOfTicks = len(ticks)
            case "plusone":
                numberOfTicks = int(limits[1] - limits[0] + 1)
            case int():
                pass
            case _:
                raise Exception("An invalid value of 'numberOfTicks' was provided.")

        tickSet = np.linspace(limits[0], limits[-1], numberOfTicks, endpoint=True)
        return tickSet


    @classmethod
    def GetXBoundaries(cls, axes):
        """
        Gets the minimum and maximum x tick marks on the x-axis.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            Axes to extract the information from.

        Returns
        -------
        yBoundries : list
            The minimim and maximum tick mark as a list.
        """
        ticks      = axes.get_xticks()
        boundries = [ticks[0], ticks[-1]]
        return boundries


    @classmethod
    def GetYBoundaries(cls, axes):
        """
        Gets the minimum and maximum y tick marks on the y-axis.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            Axes to extract the information from.

        Returns
        -------
        yBoundries : list
            The minimim and maximum tick mark as a list.
        """
        ticks      = axes.get_yticks()
        boundries = [ticks[0], ticks[-1]]
        return boundries


    @classmethod
    def SetAxesToSquare(cls, axes):
        """
        Sets the axes to have a square aspect ratio.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            axes to set the aspect ratio of.

        Returns
        -------
        None.
        """
        axes.set_aspect(1./axes.get_data_ratio())


    @classmethod
    def AddArrows(cls, axes, size=15, color="black", both=False):
        from   lendres.plotting.PlotHelper               import PlotHelper
        scale = PlotHelper.formatSettings.Scale
        axes.plot((1), (0), linestyle="", marker=">", markersize=size*scale, color=color, transform=axes.get_yaxis_transform(), clip_on=False)
        axes.plot((0), (1), linestyle="", marker="^", markersize=size*scale, color=color, transform=axes.get_xaxis_transform(), clip_on=False)

        if both:
            axes.plot((0), (0), linestyle="", marker="<", markersize=size*scale, color=color, transform=axes.get_yaxis_transform(), clip_on=False)
            axes.plot((0), (0), linestyle="", marker="v", markersize=size*scale, color=color, transform=axes.get_xaxis_transform(), clip_on=False)