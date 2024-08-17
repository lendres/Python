"""
Created on September 26, 2023
@author: lance.endres
"""
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
        self._offset                 = offset
        self._location               = location
        self._numberOfColumns        = numberOfColumns
        self._lineWidth              = lineWidth


    @property
    def Location(self):
        return self._location


    @Location.setter
    def Location(self, location:float):
        self._location = location


    @property
    def Offset(self):
        return self._offset


    @Offset.setter
    def Offset(self, offset:float):
        self._offset = offset


    @property
    def NumberOfColumns(self):
        return self._numberOfColumns


    @NumberOfColumns.setter
    def NumberOfColumns(self, numberOfColumns:int):
        if numberOfColumns < 1 or numberOfColumns > 10:
            raise Exception("Invalid number of columns specified for the legend.")
        self._numberOfColumns = numberOfColumns


    @property
    def LineWidth(self):
        return self._lineWidth


    @LineWidth.setter
    def LineWidth(self, lineWidth:float):
        self._lineWidth = lineWidth