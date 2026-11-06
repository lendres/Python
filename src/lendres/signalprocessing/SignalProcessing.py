"""
Created on March 23, 2024
@author: lance.endres
"""
from   scipy.signal                                                  import find_peaks
import heapq


class SignalProcessing():


    @classmethod
    def GetPeaks(self, y, number:int|str=6, sortBy:str="localheight", **kwargs) -> tuple[list, list]:
        """
        Finds the 'peaks'.  The peaks are the highest points in a local area.

        Parameters
        ----------
        y : sequence
            A signal with peaks.
        number : int|str optional
            The number of peaks to return or 'all'.  If 'all', all the points are returned.  The default is 6.
        sortBy : str, optional
            The method used to sort the peaks.
                localheight  : The peaks are sorted according to how high are they above the local terrain.  In this method
                    the highest peak (locally) is not necessarily the highest peak overall (globally).
                globalheight : The peaks are sorted according to which ones are highest overall (globally).
            The default is "localheight".
        **kwargs : keyword arguments
            Keyword arguments passed to the 'find_peaks' algorithm.

        Returns
        -------
        largestPeaksIndices, largestYValues : list, list
            The indices of the largest peaks and the largest peaks.
        """
        # Create default arguments, then override/update with any specified arguments.
        arguments = {"distance" : 4, "prominence" : 0.1}
        arguments.update(kwargs)

        # We find the peaks.
        # The distance argument is provided to group values that are extremely close together.  I.e., a shallow slow with small local peaks is not of interest.
        # The height argument is provided only to get the algorithm to return the relative peak prominences.  The relative heights/prominences are used as an 'importance' factor in sorting.
        # The indices of the peaks are the first firsted value from find_peaks.
        peakResults  = find_peaks(y, **arguments)

        # Extract the top values from the results.  The prominances are the local heights and the first entry returned in peakResults are the y values.
        # The output of find_peaks is [[y_values], dict{}]
        localHeights = (peakResults[1])["prominences"]
        peakIndices  = peakResults[0]

        # Handle input arguments.
        if number == "all":
            number = len(peakIndices)

        # Sort the peakIndices (absolute heights) according to their local height value (how high are they above the surrounding local territory.
        # The top values are defined as those with the largest local peak height.
        match sortBy:
            case "localheight":
                largestPeaks = heapq.nlargest(number, zip(localHeights, peakIndices))
            case "globalheight":
                largestPeaks = heapq.nlargest(number, zip(y[peakIndices], peakIndices))
            case _:
                raise Exception("The 'sortBy' parameter is not valid.")

        # Extract the y (absolute heights) from the sorted results.
        largestPeaksIndices = [peakCouple[1] for peakCouple in largestPeaks]
        largestYValues      = y[largestPeaksIndices]

        return largestPeaksIndices, largestYValues