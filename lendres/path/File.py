"""
Created on November 16, 2022
@author: Lance A. Endres
"""
import os

class File():


    @classmethod
    def SplitFileByNumberOfLines(cls, path, numberOfLines=40000, hasHeader=True):
        directory = os.path.dirname(path)

        with open(path, "r") as sourceFile:
            lineCount   = 0
            fileNumber  = 0

            header      = ""
            if hasHeader:
                header  = sourceFile.readline()
            lines       = []

            # Read all the lines.
            for line in sourceFile:
                lineCount += 1
                lines.append(line)

                # Write one file every time we reach the number of specified lines.
                if lineCount % numberOfLines == 0:
                    fileNumber += 1
                    cls._WriteFileSection(directory, fileNumber, header, lines)
                    lines = []

            # Write any remaining lines.
            if len(lines) > 0:
                fileNumber += 1
                cls._WriteFileSection(directory, fileNumber, header, lines)


    @classmethod
    def _WriteFileSection(cls, directory, fileNumber, header, lines):
        with open(directory+" part "+str(fileNumber)+".csv", "w") as outputFile:
            if header != "":
                outputFile.write(header)
            outputFile.writelines(lines)


    @classmethod
    def CombineFiles(cls, listOfFiles, outputFileName):
        if len(listOfFiles) < 2:
            raise Exception("A list of files must be supplied.")

        directory = os.path.dirname(listOfFiles[0])

        writeHeader = True

        with open(directory+outputFileName, "w") as outputFile:
            for fileName in listOfFiles:
                cls._CopyToOutputFile(fileName, outputFile, writeHeader)
                # Only write the header the from the first file.
                writeHeader = False


    @classmethod
    def _CopyToOutputFile(cls, inputFileName, outputFile, writeHeader):
        with open(inputFileName, "r") as inputFile:
            header = inputFile.readline()

            if writeHeader:
                outputFile.writelines(header)

            for line in inputFile:
                outputFile.writelines(line)