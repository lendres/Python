"""
Created on July 13, 2025
@author: Lance A. Endres
"""
import numpy                       as np
from   lendres.datatypes.ListTools import ListTools
from typing                        import Union


class LinearAlgebra():

    @classmethod
    def Normalize(cls, vector:list|tuple|np.ndarray, norm:str="L2", returnNorm:bool=False) -> Union[np.ndarray, tuple[np.ndarray, float]]:
        """
        Normalizes a 1 dimensional vector.

        Parameters
        ----------
        vector : list|tuple|np.ndarray
            The vector to normalize.
        norm : str, optional
            Normalization method.  Accepted  The default is "L2".
        returnNorm : bool, optional
            If True, the norm will be returned as a second value. The default is False.

        Raises
        ------
        Exception
            Raised if the vectors is not one dimensional.

        Returns
        -------
        np.ndarray | np.ndarray, float
            The normalized vector.  If "returnNorm" is True, the norm is also returned.
        """
        match vector:
            case list() | tuple():
                if ListTools.ContainsAtLeastOneList(vector):
                    raise Exception("This method is for one dimensional arrays only.")
                vector = np.array(vector)
            case np.ndarray():
                if vector.ndim > 1:
                    raise Exception("This method is for one dimensional arrays only.")

        normalizingValue = 0
        match norm:
            case "L1":
                normalizingValue = np.sum(np.abs(vector))
            case "L2":
                normalizingValue = np.linalg.norm(vector)
            case "max":
                normalizingValue = np.max(np.abs(vector))
            case _:
                raise Exception("Invalid norm type provided.")

        if returnNorm:
            return vector / normalizingValue, normalizingValue
        else:
            return vector / normalizingValue