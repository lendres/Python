"""
Created on December 4, 2021
@author: Lance A. Endres

The basis of this class was created in the PlotHelper.  As time went on and the PlotHelper class grew,
it was split into several classes with specific functionality.  This class resulted from that split.
"""
import numpy                                     as np
import matplotlib.pyplot                         as plt


class AxesHelper():


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
    def SetAxesToSquare(cls, axis):
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