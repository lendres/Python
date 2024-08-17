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
            offset:           float = 0.17,
            numberOfColumns:  int   = 1,
            changeLineWidths: bool  = False,
            lineWidth:        float = None,
        ):
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