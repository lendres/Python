"""
Created on November 16, 2022
@author: Lance A. Endres
"""
import os

class File():
    @classmethod
    def GetDirectory(cls, filePath):
        """
        Gets the directory from the path of a file.

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
    def GetAllFilesByExtension(cls, path, extension):
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
    def GetDirectoriesInDirectory(cls, directory):
        """
        Gets a list of directories in the specified directory.

        Parameters
        ----------
        directory : string
            The path (directory) to get the sub directories from.

        Returns
        -------
        : list of strings
            A list of directories that are subdirectories of the input path.
        """
        return [x.path for x in filter(lambda x : x.is_dir(), os.scandir(directory))]


    @classmethod
    def SplitFileByNumberOfLines(cls, path, numberOfLines=40000, hasHeader=True):
        """
        Constructor.

        Parameters
        ----------
        path : string
            Full path and file name of the file to split.
        numberOfLines : integer
            The number of lines to include in each file.
        hasHeader : bool
            Set to True if the file contains a header line (one line only).  The header will then be copied
            into each file.  Set to False if the file does not contain a header.

        Returns
        -------
        outputFiles : list
            A list of the files generated from the split.
        """
        # Removes the file extension and returns everything else, including the directory.
        baseFileName    = os.path.splitext(path)[0]

        outputFiles     = []

        with open(path, "r") as sourceFile:
            lineCount   = 0
            fileNumber  = 0

            header      = ""
            if hasHeader:
                header  = sourceFile.readline()
            lines       = []

            # Read all the lines.
            # Every time we read a "numberOfLines" a file is written from the read lines.  The number of lines
            # is the cleared and line reading continues.  Note that this means there is likely left over lines
            # as the file will not contain a number of lines evenly divisible by "numberOfLines."  Those are
            # written after the loop.
            for line in sourceFile:
                lineCount += 1
                lines.append(line)

                # Write one file every time we reach the number of specified lines.
                if lineCount % numberOfLines == 0:
                    fileNumber += 1
                    outputFileName = cls._WriteFileSection(baseFileName, fileNumber, header, lines)
                    outputFiles.append(outputFileName)
                    lines = []

            # Write any remaining lines.
            if len(lines) > 0:
                fileNumber += 1
                outputFileName = cls._WriteFileSection(baseFileName, fileNumber, header, lines)
                outputFiles.append(outputFileName)

        return outputFiles


    @classmethod
    def _WriteFileSection(cls, baseFileName, fileNumber, header, lines):
        """
        Writes the lines to a new file.

        Parameters
        ----------
        baseFileName : string
            Root file name to use for the output file name.  The part number will be appended to
            this name to create a new, unique file name.
        fileNumber : integer
            The number of the file.  This function is used when breaking a file into a sequence of files.
            This number is the current count of written files (including the current one).
        header : string
            The header text to write to the file.
        lines : list of strings
            The lines to write to the file.

        Returns
        -------
        outputFileName : string
            The path and file name of the file written.
        """
        outputFileName = baseFileName+" part "+str(fileNumber)+".csv"
        with open(outputFileName, "w") as outputFile:
            if header != "":
                outputFile.write(header)
            outputFile.writelines(lines)

        return outputFileName


    @classmethod
    def CombineFiles(cls, outputFileName, listOfFiles, removeFilesAfterCombining=False):
        """
        Combines multiple text/csv files into a single document.

        Parameters
        ----------
        outputFileName : string
            Output file path and name.
        listOfFiles : list of strings
            List of input files to read from.
        removeFilesAfterCombining : boolean
            If True, the input files are deleted after they are read.

        Returns
        -------
        None.
        """
        if len(listOfFiles) < 2:
            raise Exception("A list of files must be supplied.")

        directory   = os.path.dirname(listOfFiles[0])
        writeHeader = True

        with open(directory+"\\"+outputFileName, "w") as outputFile:
            for fileName in listOfFiles:
                # Copy the contents of the current file into the output file.
                cls._CopyToOutputFile(fileName, outputFile, writeHeader)

                # Only write the header the from the first file.
                writeHeader = False

                if removeFilesAfterCombining:
                    os.remove(fileName)


    @classmethod
    def _CopyToOutputFile(cls, inputFileName, outputFile, writeHeader):
        """
        Copies the contents of a file to the output file.  Opens the file and writes it to an existing file object.

        Parameters
        ----------
        inputFileName : string
            The path and file name of the input file to read from.
        outputFile : file object
        writeHeader : boolean
            If True, the all lines of the input file are written to the output.  If False, the first
            line of the input file is skipped and not written to the output file.

        Returns
        -------
        None.
        """
        with open(inputFileName, "r") as inputFile:
            header = inputFile.readline()

            if writeHeader:
                outputFile.writelines(header)

            lines  = []
            # Read all the lines.
            for line in inputFile:
                lines.append(line)

            outputFile.writelines(lines)