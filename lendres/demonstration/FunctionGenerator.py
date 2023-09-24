"""
Created on July 20, 2023
@author: Lance A. Endres
"""
import pandas                                    as pd
import numpy                                     as np

class FunctionGenerator():
    @classmethod
    def GetSineWave(cls, magnitude:float=10, frequency:float=4, yOffset:float=20, slope:float=0, startTime:float=0, timeLength:float=4, steps:int=1000):
        """
        Gets the X and Y values for a sine wave.

        Parameters
        ----------
         magnitude : float, optional
            Magnitude of the sine wave. The default is 10.
        frequency : float, optional
            Frequency of the sine wave. The default is 4.
        yOffset : float, optional
            A linear, vertical offset to apply to the sine wave. The default is 20.
        slope : float, optional
            A linear slope to apply to the sine wave. The default is 0.
        startTime : float, optional
            Start time (x value) for the sine wave. The default is 0.
        timeLength : float, optional
            Length of the sine wave (x length). The default is 4.
        steps : int, optional
            The number of sample points. The default is 1000.

        Returns
        -------
        x : array like
            X axis values.
        y : array like
            Y axis values.
        """
        x = np.linspace(startTime, startTime+timeLength, steps)

        # This is the angular displacement of the end point of the spring.
        y = magnitude * np.sin(2*np.pi*frequency*x)

        # Calculate the additional y values needed to account for the slope.  We want to
        # subtract out the first x value to normalize the x values and get only the contribution
        # from the slope.
        slopeOffset = [(xn-startTime)*slope for xn in x]

        # Combine the y, the linear offeset (yOffset) and offset from the slope.
        y = y + yOffset + slopeOffset

        return x, y


    @classmethod
    def GetSineWavesAsDataFrame(cls, magnitude:float|list=10, frequency:float|list=4, yOffset:float|list=20, slope:float|list=0, startTime:float=0, timeLength:float=4, steps:int=1000):
        """
        Creates a sine wave and returns it in a pandas.DataFrame.

        Parameters
        ----------
         magnitude : float, optional
            Magnitude of the sine wave. The default is 10.
        frequency : float, optional
            Frequency of the sine wave. The default is 4.
        yOffset : float, optional
            A linear, vertical offset to apply to the sine wave. The default is 20.
        slope : float, optional
            A linear slope to apply to the sine wave. The default is 0.
        startTime : float, optional
            Start time (x value) for the sine wave. The default is 0.
        timeLength : float, optional
            Length of the sine wave (x length). The default is 4.
        steps : int, optional
            The number of sample points. The default is 1000.

        Returns
        -------
        : pandas.DataFrame
        """
        match magnitude:
            case int() | float():
                x, y      = cls.GetSineWave(magnitude, frequency, yOffset, slope, startTime, timeLength, steps)
                return pd.DataFrame({"x" : x, "y" : y})
            case list():
                x, y      = cls.GetSineWave(magnitude[0], frequency[0], yOffset[0], slope[0], startTime, timeLength, steps)
                dataFrame = pd.DataFrame({"x" : x, "y0" : y})
                for i in range(1, len(magnitude)):
                    x, y  = cls.GetSineWave(magnitude[i], frequency[i], yOffset[i], slope[i], startTime, timeLength, steps)
                    dataFrame["y"+str(i)] = y
                return dataFrame


    @classmethod
    def GetMultipleSineWaveDataFrame(cls):
        """
        Creates a pandas.DataFrame with four different sine waves as data.

        Returns
        -------
        : pandas.DataFrame
        """
        x, y1  = cls.GetSineWave(9, 4, 20, 8)
        x, y2a = cls.GetSineWave(6, 2, 40)
        x, y2b = cls.GetSineWave(4, 2, 60)
        x, y3  = cls.GetSineWave(3, 1, 100)

        return pd.DataFrame({"x" : x, "y1" : y1, "y2a" : y2a, "y2b" : y2b, "y3" : y3})