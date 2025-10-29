"""
Created on October 29, 2025
@author: Lance A. Endres
"""
class StringFormat():
    @classmethod
    def CurrencyFormat(cls, showChange:bool=True, currencySymbol:str="$"):
        decimal = "0"
        if showChange:
            decimal = "2"
        return currencySymbol + "{:,." + decimal + "f}"


    @classmethod
    def Currency(cls, value:float, showChange:bool=True, currencySymbol:str="$"):
        return cls.CurrencyFormat(showChange, currencySymbol).format(value)


    @classmethod
    def FixedWidthCurrency(cls, value:float, width:int=10, showChange:bool=True, currencySymbol:str="$"):
        formatString = "{:>" + str(width) + "}"
        return formatString.format(cls.Currency(value, showChange, currencySymbol))