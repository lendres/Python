"""
Created on September 26, 2023
@author: lance.endres
"""


class LegendOptions():


    def __init__(
            self,
            location:str="outsidebottomleft",
            offset:float=0.17,
            numberOfColumns:int=1,
            changeLineWidths:bool=False,
            lineWidth:float=None,
        ):
        self.offset                 = offset
        self.location               = location
        self.numberOfColumns        = numberOfColumns
        self.lineWidth              = lineWidth


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