"""
Created on July 16, 2022
@author: Lance A. Endres
"""
import pandas                                    as pd
import numpy                                     as np
from   IPython.display                           import display

import unittest
from   lendres.ConsoleHelper                     import ConsoleHelper
from   lendres.LanguageHelper                    import LanguageHelper

import warnings

class TestLanguageHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tweets = pd.read_csv("./Data/Tweets.csv")

        #VERBOSETESTING
        #VERBOSEREQUESTED
        #cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCreditData(verboseLevel=ConsoleHelper.VERBOSETESTING, dropFirst=False)


    def setUp(self):
        """
        Set up function that runs before each test.
        """
        self.testList = ["The", ",", "and", "if", "are", "stopwords", "computer", "is", "not"]
        self.tweets   = TestLanguageHelper.tweets.copy(deep=True)


    def testStopWordsOnList(self):
        LanguageHelper.ResetStopWordsList()
        result    = LanguageHelper.RemoveStopWords(self.testList)
        self.assertEqual(result, ["The", ",", "stopwords", "computer"])

        LanguageHelper.RemoveFromStopWordsList("no")
        LanguageHelper.RemoveFromStopWordsList("not")
        result    = LanguageHelper.RemoveStopWords(self.testList)
        self.assertEqual(result, ["The", ",", "stopwords", "computer", "not"])

        LanguageHelper.AppendToStopWordsList("computer")
        result    = LanguageHelper.RemoveStopWords(self.testList)
        self.assertEqual(result, ["The", ",", "stopwords", "not"])

        result    = LanguageHelper.RemoveStopWords("The, and if are stopwords computer is not")
        self.assertEqual(result, "The, stopwords not")


    def testRemoveNumbers(self):
        result   = LanguageHelper.RemoveNumbers("Test099 on some 0.88 numbers9xx.")
        solution = "Test on some . numbersxx."
        self.assertEqual(result, solution)


    def testStripHtml(self):
        result = LanguageHelper.StripHtmlTags("<html><h2>Some important text</h2></html>")
        self.assertEqual(result, "Some important text")


    def testRemoveAccentCharacters(self):
        result = LanguageHelper.RemoveAccentedCharacters("Sómě Áccěntěd těxt")
        self.assertEqual(result, "Some Accented text")


    def testTokenize(self):
        result   = LanguageHelper.Tokenize("The , and if are stopwords computer is not")
        solution = self.testList
        self.assertEqual(result, solution)


    def testRemoveSpecialCharacters(self):
        result = LanguageHelper.RemoveSpecialCharacters("Well this was fun! What do you think? 123#@!", True)
        self.assertEqual(result, "Well this was fun What do you think")


    def testSimpleStemmer(self):
        result = LanguageHelper.SimpleStemmer("My system keeps crashing his crashed yesterday, ours crashes daily")
        self.assertEqual(result, "my system keep crash hi crash yesterday, our crash daili")


    def testLemmatize(self):
        result = LanguageHelper.Lemmatize("My system keeps crashing! his crashed yesterday, ours crashes daily")
        self.assertEqual(result, "my system keep crash ! his crash yesterday , ours crash daily")


    def testLowercase(self):
        word = "Title Case String"
        result = LanguageHelper.ToLowercase(word)
        self.assertEqual(result, "title case string")

        words = ["Title", "Case", "String"]
        result = LanguageHelper.ToLowerCase(words)
        self.assertEqual(result, ["title", "case", "string"])


    def testRemoveInternetHandles(self):
        # Test on a Pandas.Series.
        result = LanguageHelper.RemoveInternetHandles(self.tweets["text"])
        self.assertEqual(result.loc[0].strip(), "What said.")

        # Test on a string.
        result = LanguageHelper.RemoveInternetHandles(self.tweets["text"].loc[0])
        self.assertEqual(result.strip(), "What said.")

        # Test on a list.
        inputList = [self.tweets["text"].loc[0], self.tweets["text"].loc[1]]
        result = LanguageHelper.RemoveInternetHandles(inputList)
        self.assertEqual(result[0].strip(), "What said.")


    def testRemoveWebsiteAddresses(self):
        text     = self.tweets["text"].loc[21]
        result   = LanguageHelper.RemoveWebAddresses(text)
        solution = "Nice RT @VirginAmerica: Vibe with the moodlight from takeoff to touchdown. #MoodlitMonday #ScienceBehindTheExperience"
        self.assertEqual(result, solution)

        text = self.tweets["text"].loc[32]
        result = LanguageHelper.RemoveWebAddresses(text)
        solution = "@VirginAmerica  DREAM"
        self.assertEqual(result, solution)

        text = self.tweets["text"].loc[59]
        result = LanguageHelper.RemoveWebAddresses(text)
        solution = "@VirginAmerica has getaway deals through May, from $59 one-way. Lots of cool cities  #CheapFlights #FareCompare"
        self.assertEqual(result, solution)


    def testWordCloud(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The input looks more like a filename than markup. You may want to open this file and pass the filehandle into Beautiful Soup.")

            LanguageHelper.ResetStopWordsList()
            LanguageHelper.CreateWordCloud(self.tweets["text"], width=2000, height=1600)

            LanguageHelper.AppendToStopWordsList(["thank", "thanks", "plane", "flight", "flights", "im", "u"])
            LanguageHelper.CreateWordCloud(self.tweets["text"], removeStopWords=True)


if __name__ == "__main__":
    unittest.main()