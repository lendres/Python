"""
Created on September 26, 2023
@author: lance.endres
"""


class LegendOptions():
    
    
    def __init__(self, showLegend:bool=True, numberOfColumns:int=1, changeLineWidths:bool=False, lineWidth:float=4.0):
        # self.showLegend             = showLegend
        self.numberOfColumns        = numberOfColumns
        self.changeLineWidths       = changeLineWidths
        self.lineWidth              = lineWidth


    # @property
    # def ShowLegend(self):
    #     return self.showLegend
    
    
    # @ShowLegend.setter
    # def ShowLegend(self, showLegend:bool):
    #     self.showLegend = showLegend
        

    @property
    def NumberOfColumns(self):
        return self.numberOfColumns


    @NumberOfColumns.setter
    def NumberOfColumns(self, numberOfColumns:int):
        if numberOfColumns < 1 or numberOfColumns > 10:
            raise Exception("Invalid number of columns specified for the legend.")
        self.numberOfColumns = numberOfColumns


    @property
    def ChangeLineWidths(self):
        return self.changeLineWidths


    @ChangeLineWidths.setter
    def ChangeLineWidths(self, changeLineWidths:bool):
        self.changeLineWidths = changeLineWidths
    
    
    @property
    def LineWidth(self):
        return self.lineWidth
    
    
    @LineWidth.setter
    def LineWidth(self, lineWidth:float):
        self.lineWidth = lineWidth