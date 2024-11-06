"""
Created on September 26, 2023
@author: lance.endres
"""
import matplotlib
import matplotlib.pyplot                                             as plt
from   lendres.plotting.LegendOptions                                import LegendOptions

class LegendHelper():


    @classmethod
    def CreateLegend(
            cls,
            figure        : matplotlib.figure.Figure,
            axes          : matplotlib.axes.Axes,
            legendOptions : LegendOptions=LegendOptions()
        )   ->              matplotlib.legend.Legend:
        """
        Create a legend

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            Matplotlib figure to create the legend for.
        axes : matplotlib.axes.Axes
            Axes to used to reference the location.
        legendOptions : LegendOptions, optional
            Options to control the placement and look of the legend. The default is LegendOptions().

        Returns
        -------
        legend : matplotlib.legend.Legend
            The newly created legend.
        """
        if legendOptions is None:
            raise Exception("LegendOptions must be supplied.")

        match legendOptions.Location:
            case "outsidebottomleft":
                loc          = "upper left"
                bboxToAnchor = (0, -legendOptions.Offset)
            case "outsidebottomcenter":
                loc          = "upper center"
                bboxToAnchor = (0.5, -legendOptions.Offset)
            case "ousiderightcenter":
                loc          = "center left"
                bboxToAnchor = (1.0+legendOptions.Offset, 0.5)
            case _:
                raise Exception("The location argument is not valid.")

        legend = figure.legend(loc=loc, bbox_to_anchor=bboxToAnchor, ncol=legendOptions.NumberOfColumns, bbox_transform=axes.transAxes)

        if legendOptions.LineWidth is not None:
            cls.SetLegendLineWidths(legend, legendOptions.LineWidth)

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