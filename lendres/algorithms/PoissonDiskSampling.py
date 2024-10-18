"""
Created on October 19, 2024
@author: Lance Endres
"""
import numpy            as np
from   shapely.geometry import Point
import shapely
import random


class PoissonDiskSampling():


    @classmethod
    def __Distance(cls, p1, p2):
        """
        Helper function to calculate the distance between two points.
        """
        return np.linalg.norm(np.array(p1) - np.array(p2))


    @classmethod
    def __AddPoint(cls, point, samplePoints, processList, grid, minX, minY, gridSize):
        """
        Helper function to add a point to the sample points and process list.
        """
        samplePoints.append(point)
        gridX = int((point[0] - minX) / gridSize)
        gridY = int((point[1] - minY) / gridSize)
        grid[gridX][gridY] = point
        processList.append(point)


    @classmethod
    def __InNeighborhood(cls, point, grid, gridWidth, gridHeight, minX, minY, gridSize, minDistance):
        # Helper function to check if a point is in the neighborhood of existing points.
        gridX = int((point[0] - minX) / gridSize)
        gridY = int((point[1] - minY) / gridSize)
        for i in range(max(gridX - 2, 0), min(gridX + 3, gridWidth)):
            for j in range(max(gridY - 2, 0), min(gridY + 3, gridHeight)):
                neighbor = grid[i][j]
                if neighbor and cls.__Distance(point, neighbor) < minDistance:
                    return False
        return True


    @classmethod
    def Sample(cls, polygon:shapely.Polygon, minDistance:float, minDistanceFromEdge:float=None, numSamplesBeforeRejection:int=30):
        if minDistanceFromEdge is None:
            minDistanceFromEdge = minDistance

        # Function to generate Poisson Disk points.
        minX, minY, maxX, maxY = polygon.bounds

        gridSize   = minDistance / np.sqrt(2)
        gridWidth  = int(np.ceil((maxX - minX) / gridSize))
        gridHeight = int(np.ceil((maxY - minY) / gridSize))

        grid         = [[None for _ in range(gridHeight)] for _ in range(gridWidth)]
        processList  = []
        samplePoints = []

        # Add initial point.
        initialPoint = [random.uniform(minX, maxX), random.uniform(minY, maxY)]
        while not polygon.contains(Point(initialPoint)):
            initialPoint = [random.uniform(minX, maxX), random.uniform(minY, maxY)]
        cls.__AddPoint(initialPoint, samplePoints, processList, grid, minX, minY, gridSize)

        # Process points and generate new points around existing points.
        while processList:
            currentPoint = random.choice(processList)
            processList.remove(currentPoint)
            for _ in range(numSamplesBeforeRejection):
                angle    = random.uniform(0, 2 * np.pi)
                radius   = random.uniform(minDistance, 2 * minDistance)
                newPoint = [
                    currentPoint[0] + radius * np.cos(angle),
                    currentPoint[1] + radius * np.sin(angle)
                ]
                pointNewPoint = Point(newPoint)
                if polygon.contains(pointNewPoint) and cls.__InNeighborhood(newPoint, grid, gridWidth, gridHeight, minX, minY, gridSize, minDistance):
                    # Ensure it's far from the boundary.
                    if polygon.exterior.distance(pointNewPoint) >= minDistanceFromEdge:
                        cls.__AddPoint(newPoint, samplePoints, processList, grid, minX, minY, gridSize)

        return samplePoints