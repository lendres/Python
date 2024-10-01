"""
Created on August 12, 2022
@author: Lance A. Endres
"""

class Precision():
    Epsilon = 1e-10;


    @classmethod
    def IsZero(cls, value:float, epsilon:float=None):
        """
        Finite precision arthimetic zero check.  A number is considered zero if it is less than the specifiied
        perceision value (epsilon).

        Parameters
        ----------
        value : float
            The value to check.
        epsilon : float, optional
            The epsilon value to use.  If None is specified, the value from the class will be used. The default is None.

        Returns
        -------
        bool
            Returns True if the value is less than epsilon, False otherwise.
        """
        if epsilon is None:
            epsilon = cls.Epsilon

        return abs(value) < epsilon


    @classmethod
    def Equal(cls, value1:float, value2:float, epsilon:float=None):
        """
        Finite precision arthimetic check for equality.  The numbers are considered equal if their difference is
        less than the specifiied perceision value (epsilon).

        Parameters
        ----------
        value1 : float
            The first value.
        value2 : float
            The second value.
        epsilon : float, optional
            The epsilon value to use.  If None is specified, the value from the class will be used. The default is None.

        Returns
        -------
        bool
            Returns True if the difference of the values is less than epsilon, False otherwise.
        """
        if epsilon is None:
            epsilon = cls.Epsilon

        return abs(value1 - value2) < epsilon