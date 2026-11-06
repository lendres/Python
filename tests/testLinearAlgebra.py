"""
Created on July 13, 2025
@author: Lance A. Endres
"""
import numpy                                                         as np
import math

import copy

from   lendres.io.ConsoleHelper                                      import ConsoleHelper
from   lendres.mathematics.LinearAlgebra                             import LinearAlgebra

import unittest


class TestLinearAlgebra(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        verboseLevel = ConsoleHelper.VERBOSEREQUESTED
        # verboseLevel = ConsoleHelper.VERBOSETESTING
        cls.consoleHelper = ConsoleHelper(verboseLevel=verboseLevel)

        cls.list            = [1, 2, 3, 4]
        cls.tuple           = (1, 2, 3, 4)
        cls.normL1          = 1 + 2 + 3 + 4
        cls.normL2          = math.sqrt(1+4+9+16)
        cls.normMax         = 4
        cls.solutionL1      = [item / cls.normL1 for item in cls.list]
        cls.solutionL2      = [item / cls.normL2 for item in cls.list]
        cls.solutionMax     = [item / cls.normMax for item in cls.list]

        cls.listWithList    = [[1, 2], 3, 4]

        # Tuples have to have at least 2 elements.
        cls.listWithTuple   = [(1, 2), 3, 4]

        cls.array           = np.array([1, 2, 3, 4])
        cls.array2d         = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])


    def testL1NormWithList(self):
        self._TestL1(self.list)


    def testL1NormWithTuple(self):
        self._TestL1(self.tuple)


    def testL1NormWithArray(self):
        self._TestL1(np.array(self.list))


    def _TestL1(self, vector):
        result = LinearAlgebra.Normalize(vector, norm="L1")
        np.testing.assert_allclose(self.solutionL1, result)

        result, norm = LinearAlgebra.Normalize(self.list, norm="L1", returnNorm=True)
        np.testing.assert_allclose(self.solutionL1, result)
        self.assertAlmostEqual(self.normL1, norm, 5)


    def testL2NormWithList(self):
        self._TestL2(self.list)


    def testL2NormWithArray(self):
        self._TestL2(np.array(self.list))


    def _TestL2(self, vector):
        result = LinearAlgebra.Normalize(vector)
        np.testing.assert_allclose(self.solutionL2, result)

        result, norm = LinearAlgebra.Normalize(self.list, returnNorm=True)
        np.testing.assert_allclose(self.solutionL2, result)
        self.assertAlmostEqual(self.normL2, norm, 5)


    def testMaxNormWithList(self):
        self._TestMax(self.list)


    def testMaxNormWithArray(self):
        self._TestMax(np.array(self.list))


    def _TestMax(self, vector):
        result = LinearAlgebra.Normalize(vector, norm="max")
        np.testing.assert_allclose(self.solutionMax, result)

        result, norm = LinearAlgebra.Normalize(self.list, norm="max", returnNorm=True)
        np.testing.assert_allclose(self.solutionMax, result)
        self.assertAlmostEqual(self.normMax, norm, 5)


if __name__ == "__main__":
    unittest.main()