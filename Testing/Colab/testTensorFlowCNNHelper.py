"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import pandas                                                        as pd
import numpy                                                         as np

import os

import keras
from   tensorflow.keras                                              import losses
from   tensorflow.keras                                              import optimizers
from   tensorflow.keras                                              import Sequential
from   tensorflow.keras.layers                                       import Dense
from   tensorflow.keras                                              import backend
from   tensorflow.keras.layers                                       import Conv2D
from   tensorflow.keras.layers                                       import MaxPooling2D
from   tensorflow.keras.layers                                       import Flatten
from   tensorflow.keras.layers                                       import Dropout

from   lendres.io.ConsoleHelper                                      import ConsoleHelper
from   lendres.ImageDataHelper                                       import ImageDataHelper
from   lendres.TensorFlowCNNHelper                                   import TensorFlowCNNHelper


import unittest


class TestTensorFlowCNNHelper(unittest.TestCase):
    #verboseLevel = ConsoleHelper.VERBOSENONE
    #verboseLevel = ConsoleHelper.VERBOSETESTING
    verboseLevel = ConsoleHelper.VERBOSEREQUESTED
    #verboseLevel = ConsoleHelper.VERBOSEIMPORTANT


    @classmethod
    def setUpClass(cls):
        imagesInputFile = "plant-species-images-reduced.npy"
        labelsFile      = "plant-species-labels-reduced.csv"

        imagesInputFile = os.path.join("../Data", imagesInputFile)
        labelsFile      = os.path.join("../Data", labelsFile)

        consoleHelper   = ConsoleHelper(verboseLevel=cls.verboseLevel, useMarkDown=True)
        cls.dataHelper  = ImageDataHelper(consoleHelper=consoleHelper)
        cls.dataHelper.LoadImagesFromNumpyArray(imagesInputFile);
        cls.dataHelper.LoadLabelsFromCsv(labelsFile);

        cls.dataHelper.SplitData(testSize=0.2, validationSize=0.25, stratify=True)
        cls.dataHelper.NormalizePixelValues()

        cls.dataHelper.EncodeDependentVariableForAI()

        cls.inputShape  = cls.dataHelper.GetImageShape()
        cls.numberOfOutputCategories = cls.dataHelper.numberOfLabelCategories


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        self.dataHelper = TestTensorFlowCNNHelper.dataHelper.Copy()

        backend.clear_session()

        # Create model.
        self.model = Sequential()

        # Convolution and pooling layers.
        self.model.add(Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=TestTensorFlowCNNHelper.inputShape))
        self.model.add(MaxPooling2D((2, 2), padding="same"))
        self.model.add(Conv2D(8, (3, 3), activation="relu", padding="same"))
        self.model.add(MaxPooling2D((2, 2), padding="same"))

        # Flattening the output of the convolution layer after max pooling to make it ready for creating dense connections.
        self.model.add(Flatten())

        # Multiple Dense units with Relu activation.
        # Dense layer with Relu activation.
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dropout(0.2))

        # For multiclass classification Softmax is used.
        self.model.add(Dense(TestTensorFlowCNNHelper.numberOfOutputCategories, activation="softmax"))

        # Optimizer.
        adam = optimizers.Adam()

        # Loss function = categorical cross entropy.
        self.model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        # Looking into our base model.
        #self.model.summary()


    def testCreateTrainingAndValidationHistoryPlot(self):
        tensorFlowHelper = TensorFlowCNNHelper(self.dataHelper, self.model)

        tensorFlowHelper.Fit(
            epochs=6,
            verbose=TestTensorFlowCNNHelper.verboseLevel
        )

        tensorFlowHelper.Predict()

        tensorFlowHelper.CreateTrainingAndValidationHistoryPlot("loss");
        tensorFlowHelper.CreateTrainingAndValidationHistoryPlot("accuracy");


if __name__ == "__main__":
    unittest.main()