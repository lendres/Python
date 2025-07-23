"""
Created on July 13, 2025
@author: Lance A. Endres
"""
import re
import numpy                       as np
from   lendres.datatypes.ListTools import ListTools
from   typing                      import Union


class LinearAlgebra():

    @classmethod
    def Normalize(cls, vector:list|tuple|np.ndarray, norm:int|float|str = 2, returnNorm:bool=False) -> Union[np.ndarray, tuple[np.ndarray, float]]:
        """
        Normalizes a 1 dimensional vector.

        Parameters
        ----------
        vector : list|tuple|np.ndarray
            The vector to normalize.
        norm : int|float|str, optional
            Order of the norm. The default is 2 for the L2 norm.
            For the infinity norm, pass "max", "inf" or np.inf.
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

        if norm in ["max", "inf", np.inf]:
            norm = np.inf

        elif isinstance(norm, str):

            # Transforming "L1" into 1, "l2" into 2, etc.
            match = re.match(pattern=r"^L([0-9]){1}$", string=norm, flags=re.IGNORECASE)
            if match is not None:
                norm = match.groups()[0]

            try:
                norm = int(norm)
            except ValueError:
                raise Exception("Invalid norm type provided.")

        normalizingValue = np.linalg.norm(x=vector, ord=norm)

        if returnNorm:
            return vector / normalizingValue, normalizingValue
        else:
            return vector / normalizingValue