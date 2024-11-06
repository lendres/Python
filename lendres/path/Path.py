"""
Created on January 21, 2024
@author: Lance A. Endres
"""
import os
import re


class Path():

    @classmethod
    def GetDirectory(cls, filePath:str):
        """
        Gets the directory from the path of a file.  First resolves the path to be absolute.

        Parameters
        ----------
        filePath : string
            The path and file name.

        Returns
        -------
        : string
            The directory part of the file name and path.
        """
        return os.path.dirname(os.path.abspath(filePath))


    @classmethod
    def ContainsDirectory(cls, filePath:str):
        """
        Returns True if the filePath contains the directory part and False if not.

        Parameters
        ----------
        filePath : str
            A file name or path that contains the file name.

        Returns
        -------
        bool
            True if filePath contains the directory, False otherwise.
        """
        directory = os.path.dirname(filePath)
        return directory != ""


    @classmethod
    def SetWorkingDirectoryFromFilePath(cls, filePath:str):
        """
        Sets the current working directory to that of the file's directory.

        Parameters
        ----------
        filePath : str
            A file name or path that contains the file name.

        Returns
        -------
        None.
        """
        os.chdir(cls.GetDirectory(filePath))

    @classmethod
    def ChangeDirectoryDotDot(cls, path:str, levels:int=1):
        """
        Returns the directory that is "levels" up from the provided path.  Works similar to the "CD.." OOS command, but with some added features.  The path
        can be a file or directory.  If it is a file, the file name is removed and then the command is executed on the directory part.  This version also allows
        for traversing multiple levels at once.  If levels is 2, it is similar to using CD..,CD.. at the DOC command prompt.

        Parameters
        ----------
        path : str
            Path to remove levels from.  Can be a full path to a file or directory.  If it is a file, the file part is first stripped.
        levels : int, optional
            Number of durectories to go up. The default is 1.

        Returns
        -------
        : str
            The new path.
        """
        if os.path.exists(path):
            # If the path exists, determine if it is a file or directory.  We only want the directory.
            if os.path.isfile(path):
                path = cls.GetDirectory(path)

        for i in range(levels):
            path = os.path.join(path, '..')

        return os.path.abspath(path)


    @classmethod
    def GetAllFilesByExtension(cls, path:str, extension:str):
        """
        Gets a list of files that have the specified extension in the specified directory.

        Parameters
        ----------
        path : string
            The directory to search.

        Returns
        -------
        : list of string.
            List of files that have the specified extension.
        """
        filenames = os.listdir(path)
        return [filename for filename in filenames if filename.endswith(extension)]


    @classmethod
    def GetDirectoriesInDirectory(cls, directory:str):
        """
        Gets a list of directories in the specified directory.

        Parameters
        ----------
        directory : str
            The path (directory) to get the sub directories from.

        Returns
        -------
        : list of strings
            A list of directories that are subdirectories of the input path.
        """
        return [x.path for x in filter(lambda x : x.is_dir(), os.scandir(directory))]


    @classmethod
    def GetMatchingDirectories(cls, basePath, pattern):
        """
        Get all directories in the specified basePath whose names match the given pattern.

        Parameters
        ----------
        basePath : str
            The path to the subdirectory to search within.
        pattern : str
            The pattern to match directory names against.

        Returns
        -------
        matchingDirs : list
            A list of directory paths that match the pattern.
        """
        # Ensure the basePath is a directory.
        if not os.path.isdir(basePath):
            raise ValueError(f"The path {basePath} is not a directory or does not exist.")

        matchingDirs = []
        pattern = re.compile(pattern)

        # Iterate over items in the base directory.
        for item in os.listdir(basePath):
            itemPath = os.path.join(basePath, item)
            # Check if it's a directory and matches the pattern.
            if os.path.isdir(itemPath) and pattern.match(item):
                matchingDirs.append(itemPath)

        return matchingDirs