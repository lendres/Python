"""
Created on May 30, 2022
@author: Lance A. Endres
"""
import pandas                                    as pd
import numpy                                     as np

import os

from   sklearn.model_selection                   import train_test_split

import keras
from   tensorflow.keras.utils                    import to_categorical
from   tensorflow.keras                          import losses
from   tensorflow.keras                          import optimizers
from   tensorflow.keras                          import Sequential
from   tensorflow.keras.layers                   import Dense

from   keras.datasets                            import mnist

from   lendres.ConsoleHelper                     import ConsoleHelper
from   lendres.plotting.PlotMaker                import PlotMaker
from   lendres.DataHelper                        import DataHelper
from   lendres.LanguageDataHelper                import LanguageDataHelper
from   lendres.TensorFlowMultiClassHelper        import TensorFlowMultiClassHelper

import unittest


class TestTensorFlowLanguageProcessing(unittest.TestCase):
    #verboseLevel = ConsoleHelper.VERBOSENONE
    #verboseLevel = ConsoleHelper.VERBOSETESTING
    verboseLevel = ConsoleHelper.VERBOSEREQUESTED
    #verboseLevel = ConsoleHelper.VERBOSEIMPORTANT

    @classmethod
    def setUpClass(cls):
        inputFile = "preprocessed-tweets-reduced.csv"

        # Load data from the drive.
        inputFile = os.path.join("../Data", inputFile)
        data      = pd.read_csv(inputFile)

        # Create the DataHelper.
        consoleHelper   = ConsoleHelper(verboseLevel=cls.verboseLevel, useMarkDown=True)
        cls.dataHelper  = LanguageDataHelper(data=data, consoleHelper=consoleHelper)

        consoleHelper.PrintSectionTitle("Before Label Encoded", verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        consoleHelper.Display(cls.dataHelper.data.head(), verboseLevel=ConsoleHelper.VERBOSEREQUESTED)

        cls.dataHelper.LabelEncodeColumns("airline_sentiment")

        consoleHelper.PrintSectionTitle("After Label Encoded", verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        consoleHelper.Display(cls.dataHelper.data.head(), verboseLevel=ConsoleHelper.VERBOSEREQUESTED)

        # Split data.
        cls.dataHelper.SplitData("airline_sentiment", testSize=0.2, validationSize=0.25, stratify=True)

        # Do some remaining preprocessing.  Since we are going to later encode the dependent
        # variable for AI, we do not need to pre-encode the text into numerical labels.
        #cls.dataHelper.LabelEncodeColumns("airline_sentiment")
        cls.dataHelper.Vectorize(sourceColumn="text", method="tfidf", max_features=500)

        cls.dataHelper.EncodeDependentVariableForAI()
        cls.dataHelper.DisplayAIEncodingResults(numberOfEntries=5)
        cls.dataHelper.consoleHelper.PrintSectionTitle("Language Processing Test", verboseLevel=ConsoleHelper.VERBOSEREQUESTED)

    def setUp(self):
        """
        Set up function that runs before each test.  Creates a new copy of the data and uses
        it to create a new regression helper.
        """
        inputShape          = TestTensorFlowLanguageProcessing.dataHelper.GetAIInputShape()
        numberOfOutputNodes = TestTensorFlowLanguageProcessing.dataHelper.GetNumberOfUniqueCategories()

        consoleHelper = TestTensorFlowLanguageProcessing.dataHelper.consoleHelper
        consoleHelper.Display("Input shape: " + str(inputShape), verboseLevel=ConsoleHelper.VERBOSEREQUESTED)
        consoleHelper.Display("Number of output nodes: " + str(numberOfOutputNodes), verboseLevel=ConsoleHelper.VERBOSEREQUESTED)

        # Create model.
        self.model = Sequential()  

        # Multiple Dense units with Relu activation.
        self.model.add(Dense(128, activation="relu", kernel_initializer="he_uniform", input_shape=inputShape))
        self.model.add(Dense(64,  activation="relu", kernel_initializer="he_uniform"))
        self.model.add(Dense(32,  activation="relu", kernel_initializer="he_uniform"))

        # For multiclass classification Softmax is used.
        self.model.add(Dense(numberOfOutputNodes, activation="softmax"))

        # Optimizer.
        adam = optimizers.Adam(learning_rate=1e-3)

        # Loss function = categorical cross entropy.
        self.model.compile(loss=losses.categorical_crossentropy, optimizer=adam, metrics=["accuracy"])

        # Looking into our base model.
        #self.model.summary()


    def testCreateTrainingAndValidationHistoryPlot(self):
        tensorFlowHelper = TensorFlowMultiClassHelper(TestTensorFlowLanguageProcessing.dataHelper, self.model)

        tensorFlowHelper.Fit(
            epochs=6,
            verbose=1
        )

        tensorFlowHelper.Predict()

        tensorFlowHelper.CreateTrainingAndValidationHistoryPlot("loss");
        tensorFlowHelper.CreateTrainingAndValidationHistoryPlot("accuracy");


if __name__ == "__main__":
    unittest.main()