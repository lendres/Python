"""
Created on September 26, 2023
@author: lance.endres
"""
import copy
from   typing import Self

class LegendOptions():
    """
    A class that allows passing options for legend building.
    """


    def __init__(
            self,
            location:         str   = "outsidebottomleft",
            offset:           float = 0.15,
            numberOfColumns:  int   = 1,
            lineWidth:        float = None
        ):
        """
        Options to control how the LegendHelper creates legends.

        Parameters
        ----------
        location : str, optional
            Location to create the legend. The default is "outsidebottomleft".  The options are:
                outsidebottomleft
                outsidebottomcenter
                ousiderightcenter
        offset : float, optional
            The distance to offset the legend of the anchor point. The default is 0.15.
        numberOfColumns : int, optional
            Number of columns in the legend. The default is 1.
        lineWidth : float, optional
            If specified, the line widths in the legend are set to this value.  If None, the original line widths are
            kept. The default is None.

        Returns
        -------
        None.
        """
        self.offset                 = offset
        self.location               = location
        self.numberOfColumns        = numberOfColumns
        self.lineWidth              = lineWidth


    def Copy(self) -> Self:
        return copy.deepcopy(self)


    @property
    def Location(self):
        return self.location


    @Location.setter
    def Location(self, location:float):
        self.location = location


    @property
    def Offset(self):
        return self.offset


    @Offset.setter
    def Offset(self, offset:float):
        self.offset = offset


    @property
    def NumberOfColumns(self):
        return self.numberOfColumns


    @NumberOfColumns.setter
    def NumberOfColumns(self, numberOfColumns:int):
        if numberOfColumns < 1 or numberOfColumns > 10:
            raise Exception("Invalid number of columns specified for the legend.")
        self.numberOfColumns = numberOfColumns


    @property
    def LineWidth(self):
        return self.lineWidth


    @LineWidth.setter
    def LineWidth(self, lineWidth:float):
        self.lineWidth = lineWidth