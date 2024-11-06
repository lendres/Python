"""
Created on November 5, 2024
@author: Lance A. Endres
"""
import numpy as np


class CoordinateTransform2d():


    def __init__(self, translation:list=[0, 0], angle:float=0, degrees:bool=False):
        self.translation    = translation
        self.rotationMatrix = self.CreateRotation(angle, degrees)


    def CreateRotation(self, angle:float, degrees:bool=False):
        if degrees:
            angle = np.radians(angle)

        self.rotationMatrix = [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ]


    def Apply(self, points:list|tuple|np.ndarray):
        isArrayOfPoints = True

        if type(points) is list or type(points) is tuple:
            if
            points = np.array(points)