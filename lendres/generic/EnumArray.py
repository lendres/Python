"""
Created on October 20, 2025
@author: Lance A. Endres
"""
import pandas as     pd
from   enum   import IntEnum
from   typing import Generic
from   typing import TypeVar

RowEnum = TypeVar("RowEnum", bound=IntEnum)
ColEnum = TypeVar("ColEnum", bound=IntEnum)


class EnumArray(Generic[RowEnum, ColEnum]):
    def __init__(self, rowEnum:type[RowEnum], columnEnum:type[ColEnum], fillValue:str="") -> None:
        self.rowToIndex = {enumMember: index for index, enumMember in enumerate(rowEnum)}
        self.colToIndex = {enumMember: index for index, enumMember in enumerate(columnEnum)}

        self.rows = len(self.rowToIndex)
        self.cols = len(self.colToIndex)

        self.data = [[fillValue for _ in range(self.cols)] for _ in range(self.rows)]


    def __getitem__(self, key:tuple[RowEnum, ColEnum]) -> str:
        rowEnum, columnEnum = key
        return self.data[self.rowToIndex[rowEnum]][self.colToIndex[columnEnum]]


    def __setitem__(self, key:tuple[RowEnum, ColEnum], value:str) -> None:
        rowEnum, columnEnum = key
        self.data[self.rowToIndex[rowEnum]][self.colToIndex[columnEnum]] = value


    def Row(self, rowEnum:RowEnum) -> list[str]:
        return list(self.data[self.rowToIndex[rowEnum]])


    def Column(self, columnEnum:ColEnum) -> list[str]:
        colIndex = self.colToIndex[columnEnum]
        return [row[colIndex] for row in self.data]


    def Fill(self, value:str) -> None:
        for rowIndex in range(self.rows):
            for colIndex in range(self.cols):
                self.data[rowIndex][colIndex] = value


    def FillRow(self, rowEnum:RowEnum, values:list[str]) -> None:
        rowIndex = self.rowToIndex[rowEnum]

        if len(values) != self.cols:
            raise ValueError(f"Expected {self.cols} values for row, got {len(values)}")

        for colIndex in range(self.cols):
            self.data[rowIndex][colIndex] = values[colIndex]


    def FillColumn(self, columnEnum:ColEnum, values: list[str]) -> None:
        colIndex = self.colToIndex[columnEnum]

        if len(values) != self.rows:
            raise ValueError(f"Expected {self.rows} values for column, got {len(values)}")

        for rowIndex in range(self.rows):
            self.data[rowIndex][colIndex] = values[rowIndex]


    def ToDataFrame(self) -> "pd.DataFrame":
        def getLabels(grid, axis: str) -> list[str]:
            if axis == "row":
                mapping = getattr(grid, "rowToIndex", None)
                enumType = getattr(grid, "rowEnum", None)
                length = len(getattr(grid, "data", []))
                prefix = "R"
            else:
                mapping = getattr(grid, "colToIndex", None)
                enumType = getattr(grid, "columnEnum", None)
                length = len(getattr(grid, "data", [[]])[0]) if getattr(grid, "data", []) else 0
                prefix = "C"

            if mapping:
                orderedEnums = sorted(mapping, key=lambda m: mapping[m])
                return [getattr(m, "name", str(m)) for m in orderedEnums]

            if enumType:
                return [getattr(m, "name", str(m)) for m in enumType]

            return [f"{prefix}{i}" for i in range(length)]

        indexLabels  = getLabels(self, "row")
        columnLabels = getLabels(self, "col")
        dataRows = [list(row) for row in getattr(self, "data", [])]
        return pd.DataFrame(dataRows, index=indexLabels, columns=columnLabels)