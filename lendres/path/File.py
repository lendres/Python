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
        outputFileName = baseFileName+" part "+str(fileNumber)+".csv"
        with open(outputFileName, "w") as outputFile:
            if header != "":
                outputFile.write(header)
            outputFile.writelines(lines)

        return outputFileName


    @classmethod
    def CombineFiles(cls, outputFileName, listOfFiles, removeFilesAfterCombining=False):
        if len(listOfFiles) < 2:
            raise Exception("A list of files must be supplied.")

        directory = os.path.dirname(listOfFiles[0])

        writeHeader = True

        with open(directory+"\\"+outputFileName, "w") as outputFile:
            for fileName in listOfFiles:
                cls._CopyToOutputFile(fileName, outputFile, writeHeader)
                # Only write the header the from the first file.
                writeHeader = False

                if removeFilesAfterCombining:
                    os.remove(fileName)


    @classmethod
    def _CopyToOutputFile(cls, inputFileName, outputFile, writeHeader):
        with open(inputFileName, "r") as inputFile:
            header = inputFile.readline()

            if writeHeader:
                outputFile.writelines(header)

            for line in inputFile:
                outputFile.writelines(line)