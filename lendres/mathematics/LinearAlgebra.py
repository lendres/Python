"""
Created on July 13, 2025
@author: Lance A. Endres
"""
import numpy                       as np
from   lendres.datatypes.ListTools import ListTools

class LinearAlgebra():

    @classmethod
    def Normalize(cls, vector, norm="L2", returnNorm=False):
        match vector:
            case list():
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