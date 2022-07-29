"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import pandas                                    as pd
import numpy                                     as np

from   sklearn.model_selection                   import train_test_split

import keras
from   tensorflow.keras.utils                    import to_categorical
from   tensorflow.keras                          import losses
from   tensorflow.keras                          import optimizers
from   tensorflow.keras                          import Sequential
from   tensorflow.keras.layers                   import Dense

from   keras.datasets                            import mnist

from   lendres.PlotMaker                         import PlotMaker
from   lendres.DataHelper                        import DataHelper
from   lendres.TensorFlowDataHelper              import TensorFlowDataHelper
from   lendres.TensorFlowMultiClassHelper        import TensorFlowMultiClassHelper

import unittest


class TestTensorFlowHelper(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.verboseLevel = 1

        # Get mist data.
        print("\n\nDownloading data...")
        (xTrainingData, yTrainingData), (xTestingData, yTestingData) = mnist.load_data()
        print("\nX training shape:", xTrainingData.shape)
        print("\nY training shape:", yTrainingData.shape)

        # Flatten the images.
        print("Flatten images...")
        image_vector_size = 28*28
        xTrainingData = xTrainingData.reshape(xTrainingData.shape[0], image_vector_size)
        xTestingData  = xTestingData.reshape(xTestingData.shape[0],   image_vector_size)
        print("\nX training shape:", xTrainingData.shape)

        # Normalize inputs from 0-255 to 0-1.
        print("Normalize inputs...")
        xTrainingData = xTrainingData / 255.0
        xTestingData  = xTestingData  / 255.0
        print("\nX training shape:", xTrainingData.shape)

        xTrainingData, xValidationData, yTrainingData, yValidationData = train_test_split(xTrainingData, yTrainingData, test_size=0.2, random_state=1, stratify=yTrainingData)

        # Convert to "one-hot" vectors using the to_categorical function
        cls.num_classes  = 10

        cls.dataHelper                 = TensorFlowDataHelper()
        cls.dataHelper.xTrainingData   = xTrainingData
        cls.dataHelper.xValidationData = xValidationData
        cls.dataHelper.xTestingData    = xTestingData

        cls.dataHelper.yTrainingData   = yTrainingData
        cls.dataHelper.yValidationData = yValidationData
        cls.dataHelper.yTestingData    = yTestingData

        cls.dataHelper.EncodeDependentVariable()


    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        image_size = 28*28

        # Create model.
        self.model = Sequential()  

        # Multiple Dense units with Relu activation.
        self.model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', input_shape=(image_size,)))
        self.model.add(Dense(64,  activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(32,  activation='relu', kernel_initializer='he_uniform'))

        # For multiclass classification Softmax is used.
        self.model.add(Dense(TestTensorFlowHelper.num_classes, activation='softmax'))

        # Optimizer.
        adam = optimizers.Adam(learning_rate=1e-3)

        # Loss function = categorical cross entropy.
        self.model.compile(loss=losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

        # Looking into our base model.
        #self.model.summary()


    def testCreateTrainingAndValidationHistoryPlot(self):
        tensorFlowHelper = TensorFlowMultiClassHelper(TestTensorFlowHelper.dataHelper, self.model)

        tensorFlowHelper.Fit(
            saveHistory=True,
            epochs=6,
            verbose=TestTensorFlowHelper.verboseLevel
        )

        tensorFlowHelper.Predict()

        tensorFlowHelper.CreateTrainingAndValidationHistoryPlot("loss");
        tensorFlowHelper.CreateTrainingAndValidationHistoryPlot("accuracy");


if __name__ == "__main__":
    unittest.main()