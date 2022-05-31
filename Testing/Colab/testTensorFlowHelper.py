"""
Created on May 30, 2022
@author: Lance
"""
import pandas                  as pd
import numpy                   as np

import keras
from   tensorflow.keras.utils  import to_categorical
from   tensorflow.keras        import losses
from   tensorflow.keras        import optimizers
from   tensorflow.keras        import Sequential
from   tensorflow.keras.layers import Dense

from   keras.datasets          import mnist

from lendres.PlotMaker         import PlotMaker
from lendres.TensorFlowHelper  import TensorFlowHelper

import unittest

class TestTensorFlowHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (cls.x_train, cls.y_train), (cls.x_test, cls.y_test) = mnist.load_data()

        # Flatten the images
        image_vector_size = 28*28
        cls.x_train = cls.x_train.reshape(cls.x_train.shape[0], image_vector_size)
        cls.x_test  = cls.x_test.reshape(cls.x_test.shape[0],   image_vector_size)

        # # normalize inputs from 0-255 to 0-1
        cls.x_train = cls.x_train / 255.0
        cls.x_test  = cls.x_test  / 255.0

        # Convert to "one-hot" vectors using the to_categorical function
        cls.num_classes  = 10
        cls.y_train      = to_categorical(cls.y_train, cls.num_classes)
        cls.y_test_cat   = to_categorical(cls.y_test,  cls.num_classes)

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
        self.model.add(Dense(64, activation='relu',  kernel_initializer='he_uniform'))
        #self.model.add(Dense(64, activation='relu',  kernel_initializer='he_uniform'))
        self.model.add(Dense(32, activation='relu',  kernel_initializer='he_uniform'))

        # For multiclass classification Softmax is used.
        self.model.add(Dense(TestTensorFlowHelper.num_classes, activation='softmax'))

        # Optimizer.
        adam = optimizers.Adam(lr=1e-3)

        # Loss function = categorical cross entropy.
        self.model.compile(loss=losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

        # Looking into our base model.
        #self.model.summary()


    def testCreateTrainingAndValidationHistoryPlot(self):
        history = self.model.fit(
            TestTensorFlowHelper.x_train,
            TestTensorFlowHelper.y_train,
            validation_split=0.2,
            epochs=8,
            batch_size=128,
            verbose=2
        )
        TensorFlowHelper.CreateTrainingAndValidationHistoryPlot(history, "loss")


if __name__ == "__main__":
    unittest.main()