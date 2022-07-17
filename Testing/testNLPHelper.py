"""
Created on July 16, 2022
@author: Lance A. Endres
"""
import numpy                                     as np
from   IPython.display                           import display

import unittest
from   lendres.ConsoleHelper                     import ConsoleHelper
from   lendres.NLPHelper                         import NLPHelper

class TestNLPHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.regresionHelpers = []

        #VERBOSETESTING
        #VERBOSEREQUESTED
        #cls.dataHelper, cls.dependentVariable = DataSetLoading.GetCreditData(verboseLevel=ConsoleHelper.VERBOSETESTING, dropFirst=False)


    def setUp(self):
        """
        Set up function that runs before each test.
        """
        self.testList = ["The", ",", "and", ",", "if", "are", "stopwords", ",", "computer", "is", "not"]


    def testStopWords(self):
        nlpHelper = NLPHelper()
        nlpHelper.RemoveStopWords("no")
        nlpHelper.RemoveStopWords("not")
        result    = nlpHelper.FilterOutStopWords(self.testList)
        self.assertEqual(result, ["The", ",", ",", "stopwords", ",", "computer", "not"])


    def testStripHtml(self):
        result = NLPHelper.StripHtmlTags("<html><h2>Some important text</h2></html>")
        self.assertEqual(result, "Some important text")


    def testRemoveAccentCharacters(self):
        result = NLPHelper.RemoveAccentedCharacters("Sómě Áccěntěd těxt")
        self.assertEqual(result, "Some Accented text")


    def testTokenize(self):
        result   = NLPHelper.Tokenize("The , and , if are stopwords, computer is not")
        solution = self.testList
        self.assertEqual(result, solution)


    def testRemoveSpecialCharacters(self):
        result = NLPHelper.RemoveSpecialCharacters("Well this was fun! What do you think? 123#@!", True)
        self.assertEqual(result, "Well this was fun What do you think ")


    def testSimpleStemmer(self):
        result = NLPHelper.SimpleStemmer("My system keeps crashing his crashed yesterday, ours crashes daily")
        self.assertEqual(result, "my system keep crash hi crash yesterday, our crash daili")


    def testLemmatize(self):
        result = NLPHelper.Lemmatize("My system keeps crashing! his crashed yesterday, ours crashes daily")
        self.assertEqual(result, "my system keep crash ! his crash yesterday , ours crash daily")

    def testLowercase(self):
        word = "Title Case String"
        result = NLPHelper.ToLowerCase(word)
        self.assertEqual(result, "title case string")

        words = ["Title", "Case", "String"]
        result = NLPHelper.ToLowerCase(words)
        self.assertEqual(result, ["title", "case", "string"])


if __name__ == "__main__":
    unittest.main()