"""
Created on October 1, 2023
@author: Lance A. Endres
"""
import cv2

import DataSetLoading
from   lendres.plotting.PlotHelper                                   import PlotHelper
from   lendres.computervision.ImageHelper                            import ImageHelper

import unittest


class TestImageHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        PlotHelper.PushSettings(parameterFile=ImageHelper.DefaultSettings())

        # Load the data.
        filePath      = DataSetLoading.GetFileInDataDirectory("logo.png")
        cls.image     = cv2.imread(filePath)
        # cls.imageData = np.array(image, dtype=float)


    def testImageHelperPlot(self):
        ImageHelper.PlotImage(self.image, "ImageHelper Plot")
        # cv2.COLOR_BGR2GRAY


    def testHighPassFilter(self):
        result = ImageHelper.ApplyHighPassFilter(self.image, ksize=(21, 21), sigmaX=3)
        ImageHelper.PlotImage(result, "High Pass Filter")


    @classmethod
    def tearDownClass(cls):
        PlotHelper.PopSettings()


if __name__ == "__main__":
    unittest.main()