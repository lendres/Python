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

class TestNLPHelper(unittest.TestCase):

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
        self.testList = ["The", ",", "and", ",", "if", "are", "stopwords", ",", "computer", "is", "not"]
        self.tweets   = TestNLPHelper.tweets.copy(deep=True)


    def testStopWords(self):
        LanguageHelper.RemoveStopWords("no")
        LanguageHelper.RemoveStopWords("not")
        result    = LanguageHelper.FilterOutStopWords(self.testList)
        self.assertEqual(result, ["The", ",", ",", "stopwords", ",", "computer", "not"])


    def testStripHtml(self):
        result = LanguageHelper.StripHtmlTags("<html><h2>Some important text</h2></html>")
        self.assertEqual(result, "Some important text")


    def testRemoveAccentCharacters(self):
        result = LanguageHelper.RemoveAccentedCharacters("Sómě Áccěntěd těxt")
        self.assertEqual(result, "Some Accented text")


    def testTokenize(self):
        result   = LanguageHelper.Tokenize("The , and , if are stopwords, computer is not")
        solution = self.testList
        self.assertEqual(result, solution)


    def testRemoveSpecialCharacters(self):
        result = LanguageHelper.RemoveSpecialCharacters("Well this was fun! What do you think? 123#@!", True)
        self.assertEqual(result, "Well this was fun What do you think ")


    def testSimpleStemmer(self):
        result = LanguageHelper.SimpleStemmer("My system keeps crashing his crashed yesterday, ours crashes daily")
        self.assertEqual(result, "my system keep crash hi crash yesterday, our crash daili")


    def testLemmatize(self):
        result = LanguageHelper.Lemmatize("My system keeps crashing! his crashed yesterday, ours crashes daily")
        self.assertEqual(result, "my system keep crash ! his crash yesterday , ours crash daily")


    def testLowercase(self):
        word = "Title Case String"
        result = LanguageHelper.ToLowerCase(word)
        self.assertEqual(result, "title case string")

        words = ["Title", "Case", "String"]
        result = LanguageHelper.ToLowerCase(words)
        self.assertEqual(result, ["title", "case", "string"])


    def testStripHandles(self):
        # Test on a Pandas.Series.
        result = LanguageHelper.StripHandles(self.tweets["text"])
        self.assertEqual(result.loc[0].strip(), "What said.")

        # Test on a string.
        result = LanguageHelper.StripHandles(self.tweets["text"].loc[0])
        self.assertEqual(result.strip(), "What said.")

        # Test on a list.
        inputList = [self.tweets["text"].loc[0], self.tweets["text"].loc[1]]
        result = LanguageHelper.StripHandles(inputList)
        self.assertEqual(result[0].strip(), "What said.")



if __name__ == "__main__":
    unittest.main()