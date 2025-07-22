"""
Created on July 16, 2025
@author: Lance A. Endres
"""
import os
import sys

from   lendres.path.Path            import Path


class System():

    @classmethod
    def AppendPath(cls, path:str):
        """
        Appends the path to the system path.  Checks that the path does not already exist before adding it.

        Parameters
        ----------
        path : string
            The to append to the system path.

        Returns
        -------
        None.
        """
        if not cls.DoesPathExistInSysPath(path):
            # Add the current directory to sys.path.
            sys.path.append(path)


    @classmethod
    def AppendPathRelativeToFileLocation(cls, filePath:str, levelsToMoveUp:int=None, appendPath:str=None):
        """
        Allows for appending a directory relative to a file location.  This first extracts the directory from the file,
        then goes "up" by the number of directories specified (if any), and then appends a path (if provided).
        Checks that the path does not already exist before adding it.

        Parameters
        ----------
        filePath : string
            The path of a file.

        Returns
        -------
        None.
        """
        path = os.path.dirname(filePath)

        if levelsToMoveUp is not None:
            path = Path.ChangeDirectoryDotDot(path, levelsToMoveUp)

        if appendPath is not None:
            path = os.path.join(path, appendPath)

        cls.AppendPath(path)


    @classmethod
    def DoesPathExistInSysPath(cls, path:str) -> bool:
        """
        Checks if a path exists in the sys.path.

        Parameters
        ----------
        path : str
            Path to look for.

        Returns
        -------
        bool
            True if the path already exists in sys.path, False, otherwise.
        """
        # Normalizes the paths so that comparisons are accurate.
        normalizedPath = os.path.abspath(path)
        return any(os.path.abspath(existingPath) == normalizedPath for existingPath in sys.path)