"""
Created on June 24, 2022
@author: Lance A. Endres
"""
from   matplotlib                                import pyplot                     as plt
import math
import cv2

from   lendres.PlotHelper                        import PlotHelper

class ImageHelper():
    arrayImageSize = 2.5


    @classmethod
    def PlotImage(cls, image, title=None, size=6, colorConversion=None):
        """
        Plot example image.

        Parameters
        ----------
        image : image
            Image to plot.
        title : string
            Title of the figure.
        size : float
            Size (width and height) of figure.
        colorConversion : OpenCV color conversion enumeration.
            Color conversion to perform before plotting.  Images are plotted in RGB.  For example, if the
            image is in BGR cv2.COLOR_BGR2RGB should be passed.

        Returns
        -------
        None.
        """
        # Defining the figure size.  Automatically adjust for the number of images to be displayed.
        #PlotHelper.scale = 0.65
        PlotHelper.FormatPlot(width=size, height=size)

        # Adding subplots with 3 rows and 4 columns.
        axis = plt.gca()

        # Plotting the image.
        if colorConversion != None:
            image = cv2.cvtColor(image, colorConversion)
        axis.imshow(image)

        if title != None:
            axis.set_title(title)

        # Turn off white grid lines.
        plt.grid(False)

        plt.show()
        PlotHelper.scale = 1.0

    @classmethod
    def CreateImageArrayPlot(cls, images, labels, columns=4, colorConversion=None):
        """
        Plots the images in an array.

        Parameters
        ----------
        images : array like
            Set of images to plot.
        labels : array like
            Set of labels to use for the individual images.
        columns : integer
            The number of columns to plot.
        colorConversion : OpenCV color conversion enumeration.
            Color conversion to perform before plotting.  Images are plotted in RGB.  For example, if the
            image is in BGR cv2.COLOR_BGR2RGB should be passed.


        Returns
        -------
        None.
        """
        # Calculate required values.
        numberOfImages = len(images)
        rows = math.ceil(numberOfImages / columns)

        # Defining the figure size.  Automatically adjust for the number of images to be displayed.
        PlotHelper.scale = 0.55
        PlotHelper.FormatPlot(width=columns*ImageHelper.arrayImageSize+2, height=rows*ImageHelper.arrayImageSize+2)
        figure = plt.figure()

        # Position in the index array/range.
        k = -1

        for i in range(columns):
            for j in range(rows):
                # Adding subplots with 3 rows and 4 columns.
                axis = figure.add_subplot(rows, columns, i*rows+j+1)

                # Plot the image.  Convert colors if required.
                k +=1
                image = images[k]
                if colorConversion != None:
                    image = cv2.cvtColor(image, colorConversion)
                axis.imshow(image)

                # Turn off white grid lines.
                axis.grid(False)

                axis.set_title(labels[k], y=0.9)

        # Adjust spacing so titles don't run together.
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

        plt.show()
        PlotHelper.scale = 1.0


    @classmethod
    def DisplayColorChannels(cls, image, colorConversion=None):
        """
        Displays an image along with the individual color channels.

        Parameters
        ----------
        image : image
            Image in an array.
        colorConversion : OpenCV color conversion enumeration.
            Color conversion to perform before plotting.  Images are plotted in RGB.  For example, if the
            image is in BGR cv2.COLOR_BGR2RGB should be passed.

        Returns
        -------
        None.
        """
        imageArray = cv2.split(image)
        imageArray.insert(0, image)
        titles = ["Original", "Blue", "Green", "Red"]
        ImageHelper.CreateImageArrayPlot(imageArray, titles, columns=4, colorConversion=colorConversion)


    @classmethod
    def DisplayChromaKey(cls, image, lowerBounds, upperBounds, maskBlurSize=3, colorConversion=None, inputBoundsFormat="hsv"):
        """
        Displays an image along with the image separated into two components based on chroma keying.

        Parameters
        ----------
        image : image
            Image in an array.
        lowerBounds : numpy array of 3 values.
            Lower bounds of mask.
        maskBlurSize : int
            Size of the blur to apply to the mask.  Must be an odd number.
        upperBounds : numpy array of 3 values.
            Upper bounds of mask.
        colorConversion : OpenCV color conversion enumeration.
            Color conversion to perform before plotting.  Images are plotted in RGB.  For example, if the
            image is in BGR cv2.COLOR_BGR2RGB should be passed.
        inputBoundsFormat : string
            Format of lowerBounds and upperBounds.

        Returns
        -------
        None.
        """
        imageArray = ImageHelper.ChromaKey(image, lowerBounds, upperBounds, maskBlurSize, inputBoundsFormat)

        imageArray.insert(0, image)
        titles = ["Original", "Masked Image", "Image Remainder", "Mask"]
        ImageHelper.CreateImageArrayPlot(imageArray, titles, columns=4, colorConversion=colorConversion)


    @classmethod
    def ChromaKey(cls, image, lowerBounds, upperBounds, maskBlurSize=3, inputBoundsFormat="hsv"):
        """
        Splits the image into two components based on chroma keying.

        Parameters
        ----------
        image : image
            Image in an array.
        lowerBounds : numpy array of 3 values.
            Lower bounds of mask.
        upperBounds : numpy array of 3 values.
            Upper bounds of mask.
        maskBlurSize : int
            Size of the blur to apply to the mask.  Must be an odd number.
        inputBoundsFormat : string
            Format of lowerBounds and upperBounds.

        Returns
        -------
        maskedImage : image
            Part of the image that passes the mask.
        imageRemainder : image
            Part of the image that did not pass the mask.
        mask : image
            Mask used on the image.
        """
        imageArray = []
        if inputBoundsFormat == "bgr":
            imageArray = ImageHelper.ChromaKeyWithBGR(image, lowerBounds, upperBounds, maskBlurSize)
        elif inputBoundsFormat == "hsv":
            imageArray = ImageHelper.ChromaKeyWithHSV(image, lowerBounds, upperBounds, maskBlurSize)
        else:
            raise Exception("Input bounds format argument not valid.")

        return imageArray


    @classmethod
    def ChromaKeyWithBGR(cls, image, lowerBounds, upperBounds, maskBlurSize=3):
        """
        Splits the image into two components based on chroma keying.

        Parameters
        ----------
        image : image
            Image in an array.
        lowerBounds : numpy array of 3 values.
            Lower bounds of mask.
        upperBounds : numpy array of 3 values.
            Upper bounds of mask.
        maskBlurSize : int
            Size of the blur to apply to the mask.  Must be an odd number.

        Returns
        -------
        maskedImage : image
            Part of the image that passes the mask.
        imageRemainder : image
            Part of the image that did not pass the mask.
        mask : image
            Mask used on the image.
        """
        mask           = cv2.inRange(image, lowerBounds, upperBounds)
        mask           = cv2.medianBlur(mask, maskBlurSize)
        maskedImage    = cv2.bitwise_and(image, image, mask=mask)
        imageRemainder = image - maskedImage

        return [maskedImage, imageRemainder, mask]


    @classmethod
    def ChromaKeyWithHSV(cls, image, lowerBounds, upperBounds, maskBlurSize=3):
        """
        Splits the image into two components based on chroma keying.

        Parameters
        ----------
        image : image
            Image in an array.
        lowerBounds : numpy array of 3 values.
            Lower bounds of mask.
        upperBounds : numpy array of 3 values.
            Upper bounds of mask.
        maskBlurSize : int
            Size of the blur to apply to the mask.  Must be an odd number.

        Returns
        -------
        maskedImage : image
            Part of the image that passes the mask.
        imageRemainder : image
            Part of the image that did not pass the mask.
        mask : image
            Mask used on the image.
        """
        hsvImage       = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask           = cv2.inRange(hsvImage, lowerBounds, upperBounds)
        mask           = cv2.medianBlur(mask, maskBlurSize)
        maskedImage    = cv2.bitwise_and(image, image, mask=mask)
        imageRemainder = image - maskedImage

        return [maskedImage, imageRemainder, mask]