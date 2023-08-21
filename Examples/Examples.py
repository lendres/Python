"""
Created on July 31, 2022
@author: Lance A. Endres
"""

"""
Miscellaneous Python examples.
"""

import math
import os
import random
import re
import sys


def CountNumberOfPairs(n, ar):
    # Get unique values.  Convert to list because set doesn't have a count function.# Get unique values.
    colors = list(set(ar))
    print("\nColors:")
    print(colors)

    # Count the number of times each value appears in the input.
    counts = [ar.count(i) for i in colors]
    print("\nCounts")
    print(counts)

    # Calculate the number of pairs of each type.
    pairs = [math.floor(i/2.0) for i in counts]
    print("\nPairs:")
    print(pairs)

    # Total number of pairs in the input.
    result = sum(pairs)
    print("Result:", result)
    return result


def CountingValleys(steps, path):
    elevation        = 0
    valleyCount      = 0
    inValley         = False

    for i in range(steps):
        # Track our current elevation.
        if path[i] == "U":
            elevation += 1
        else:
            elevation -= 1
        print("\nElevation:", elevation)

        # Valleys must be below sea level.
        # The problem statement is poorly worded (no suprise for HackerRank).
        # What they intend to mean is a valley is any sequence of steps that
        # start with a step down from sea level and end with a step up to sea level.
        # The way the question is worded ("consecutive steps below sea level") could
        # be interpretued to mean you need multiple steps below sea level.  At which
        # point you need to define the elevation of a step as a step starts at one
        # elevation and ends at another.
        if elevation < 0:
            inValley = True
            print("In valley")

        # This catches when we were below sea leven and return to sea level.
        if elevation == 0 and inValley:
            valleyCount += 1
            inValley    = False

    return valleyCount


def OrganizingContainers(container):
    # Sums and sorts rows.
    r = sorted([sum(x) for x in container])

    # Sums and sorts columns.
    c = sorted([sum(x) for x in zip(*container)])
    return "Possible" if r == c else "Impossible"
    # So in order to explain that, you have to understand the zip() function itself. If zip()
    # was to be given two or more arrays as arguments, it would create one multi-dimensional
    # array by grouping up the values of the given arrays by their indices.  When paired with
    # the zip() function, the "*" operator instead allows you to invert the operation and "unzip"
    # a multi-dimensional array into the component arrays it would have taken in a hypothetical
    # initial "zip()" argument. In this case, you break the container down into smaller arrays of
    # its components that previously shared the same column.  So the R operation line takes the
    # sums of the rows, and the C operation line takes the sums of the columns.



def BomberMan(r, c, n, gridString):
    time      = 1
    bombTimes = [[0]*c for i in range(r)]

    # Grid string is an array of strings like:
    # ["xoox", "xxox", "oxxo", "xoxo"]
    # This converts it to a list of lists (the strings are parsed by each character).
    grid      = [list(x) for x in gridString]

    printDebug = 0
    if printDebug:
        #print("\nInitial grid:", gridString)
        #print("\nConverted grid:", grid)
        #print("\nBomb times initial:", bombTimes)
        print("\nTotal time:", n)

    finalTime = n if n < 4 else 4 + n % 4

    while time < finalTime:
        # First second do nothing.
        # Second second plant bombs
        time += 1
        if time % 2 == 0:
            PlantBombs(r, c, n, grid, time, bombTimes, printDebug)
        else:
            ExplodeBombs(r, c, n, grid, time, bombTimes, printDebug)

    newGrid = JoinStrings(grid)
    #if printDebug:
    #    print("\nNew grid:", newGrid)
    return newGrid


def PlantBombs(r, c, n, grid, time, bombTimes, printDebug):
    #DisplayGrid("Before planting bombs", grid, printDebug)
    for i in range(r):
        for j in range(c):
            if grid[i][j] == ".":
                grid[i][j]      = "O"
                bombTimes[i][j] = time

    #DisplayGrid("After planting bombs", grid, printDebug)
    #if printDebug:
    #    print("Times:", bombTimes)


def ExplodeBombs(r, c, n, grid, time, bombTimes, printDebug):
    #DisplayGrid("Before exploding bombs", grid, printDebug)
    explodeTime = time - 3
    for i in range(r):
        for j in range(c):
            if bombTimes[i][j] == explodeTime:
                grid[i][j] = "."
                if i-1 >= 0:
                    grid[i-1][j] = "."
                if i+1 < r:
                    grid[i+1][j] = "."
                if j-1 >= 0:
                    grid[i][j-1] = "."
                if j+1 < c:
                    grid[i][j+1] = "."
    #DisplayGrid("After exploding bombs", grid, printDebug)


def JoinStrings(grid):
    return ["".join(string) for string in grid]


def DisplayGrid(label, grid, printDebug):
    if printDebug:
        print("\n"+label+":")
        joinedGrid = JoinStrings(grid)
        print('\n'.join(joinedGrid))


def BirthdayCakeCandles(candles):
    maxValue    = max(candles)
    numberOfMax = candles.count(maxValue)

    print("Candles:", candles)
    print("numberOfMax:", numberOfMax)
    return numberOfMax