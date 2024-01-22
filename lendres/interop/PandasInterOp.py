"""
Created on January 21, 2024
@author: Lance A. Endres
"""

# import pint
# import pint_pandas

# from pint import UnitRegistry
# ureg = UnitRegistry()

# @ureg.wraps(None, None)
# def GetMagnitude2(series):
#     return series


class PandasInterOp():

    @classmethod
    def GetSeriesMagnitudes(cls, series):
        """
        Determines if a series contains data that is a pint data type.  If it is, it extracts the magnitudes and
        returns those.  Otherwise the original data is returned.

        Parameters
        ----------
        series : pandas.Series
            A Pandas series.

        Returns
        -------
        pandas.Series or numpy.ndarray
            The original data or the extract magnitudes.
        """
        if str(series.dtype).startswith("pint"):
            return series.values.quantity.magnitude
        else:
            return series