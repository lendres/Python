"""
Created on September 26, 2023
@author: lance.endres
"""
import matplotlib
import matplotlib.pyplot                                             as plt
from   lendres.plotting.LegendOptions                                import LegendOptions

class LegendHelper():


    @classmethod
    def CreateLegendAtFigureBottom(cls, figure:matplotlib.figure.Figure, axes:matplotlib.axes.Axes, offset=0.15, legendOptions:LegendOptions=LegendOptions()):
        legend = None

        if legendOptions is not None:
            legend = figure.legend(loc="upper left", bbox_to_anchor=(0, -offset), ncol=legendOptions.NumberOfColumns, bbox_transform=axes.transAxes)

            if legendOptions.ChangeLineWidths:
                cls.SetLegendLineWidths(legend, legendOptions.lineWidth)

        return legend


    @classmethod
    def CreateLegendAtFigureRight(cls, figure:matplotlib.figure.Figure, axes:matplotlib.axes.Axes, offset=0.15, legendOptions:LegendOptions=LegendOptions()):
        legend = None

        if legendOptions is not None:
            legend = figure.legend(loc="center left", bbox_to_anchor=(1.0+offset, 0.5), ncol=legendOptions.NumberOfColumns, bbox_transform=axes.transAxes)

            if legendOptions.ChangeLineWidths:
                cls.SetLegendLineWidths(legend, legendOptions.lineWidth)

        return legend


    @classmethod
    def SetLegendLineWidths(cls, legend:plt.legend, linewidth:float=4.0):
        """
        Change the line width for the legend.  Sets all the line widths to the same value.  Useful for when the
        legend lines are too thin to see the color well.

        Parameters
        ----------
        legend : matplotlib.pyplot.legend
            The legend.
        linewidth : float, optional
            The line width. The default is 4.0.

        Returns
        -------
        None.
        """
        # Loop over all the lines in the legend and set the line width.  Doesn't change patches.
        for line in legend.get_lines():
            line.set_linewidth(linewidth)