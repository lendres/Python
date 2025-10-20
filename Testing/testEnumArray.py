"""
Created on November 16, 2022
@author: Lance A. Endres
"""
import os
from   enum              import IntEnum
from   enum              import auto
from   enum              import unique

from   lendres.path.Path import Path
from   lendres.generic.EnumArray import EnumArray

import DataSetLoading

import unittest

@unique
class Stock(IntEnum):
    QQQ = auto()
    SPX = auto()
    KO  = auto()
    IMB = auto()

@unique
class Analyst(IntEnum):
    Oppenheimer = auto()
    Citi        = auto()
    WellsFargo  = auto()


class TestEnumArray(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = EnumArray[Stock, Analyst](Stock, Analyst)
        cls.data.FillRow(Stock.QQQ, ["Buy", "Strong Buy", "Buy"])
        cls.data.FillRow(Stock.SPX, ["Buy", "Hold", "Hold"])
        cls.data.FillRow(Stock.KO,  ["Sell", "Hold", "Hold"])
        cls.data.FillRow(Stock.IMB, ["Sell", "Sell", "Sell"])


    def testChangeDirectoryDotDot(self):
        print(self.data.ToDataFrame().head())
        #self.assertEqual(solution, result)



if __name__ == "__main__":
    unittest.main()