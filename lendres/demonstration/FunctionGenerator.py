"""
Created on July 20, 2023
@author: Lance A. Endres
"""
import pandas                                    as pd
import numpy                                     as np

class FunctionGenerator():
    @classmethod
    def GetSineWave(cls, magnitude=10, frequency=4, yOffset=20, slope=0, startTime=0, timeLength=4, steps=1000):
        """
        Gets the X and Y values for a sine wave.

        Returns
        -------
        x : array like
            X axis values.
        y : array like
            Y axis values.
        """
        x     = np.linspace(startTime, startTime+timeLength, steps)

        # This is the angular displacement of the end point of the spring.
        y = magnitude * np.sin(2*np.pi*frequency*x)


        slope = [(xn-startTime)*slope for xn in x]

        y = y + yOffset + slope

        return x, y


    @classmethod
    def GetSineWaveDataFrame(cls):
        """
        Creates a pandas.DataFrame with different sine waves as data.

        Returns
        -------
        pandas.DataFrame
        """
        x, y1  = cls.GetSineWave()
        x, y2a = cls.GetSineWave(6, 2, 40)
        x, y2b = cls.GetSineWave(4, 2, 60)
        x, y3  = cls.GetSineWave(3, 1, 100)

        return pd.DataFrame({"x" : x, "y1" : y1, "y2a" : y2a, "y2b" : y2b, "y3" : y3})