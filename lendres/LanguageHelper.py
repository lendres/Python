"""
Created on July 16, 2022
@author: Lance A. Endres
"""

import pandas                                    as pd
import unicodedata
from   bs4                                       import BeautifulSoup

# Natural language processing tool-kit
import nltk
from   nltk.tokenize.toktok                      import ToktokTokenizer
import contractions

#for plotting images & adjusting colors
import matplotlib.pyplot                         as plt
from   wordcloud                                 import WordCloud
from   wordcloud                                 import STOPWORDS
from   wordcloud                                 import ImageColorGenerator

import re
import spacy

from   IPython.display                           import display

import lendres
from   lendres.ConsoleHelper                     import ConsoleHelper


def Install(cls):
   nltk.download("stopwords")


class LanguageHelper():
    stopWords = nltk.corpus.stopwords.words("english")


    @classmethod
    def RemoveFromStopWordsList(cls, words):
        """
        Remove words from the list of stop words.

        Parameters
        ----------
        words : string or list of strings
            Words to remove from the stop words.

        Returns
        -------
        None.
        """
        if type(words) != list:
            words = [words]

        for word in words:
            # Prevent an error or trying to remove a word not in the list.
            if word in cls.stopWords:
                cls.stopWords.remove(word)


    @classmethod
    def AppendToStopWordsList(cls, words):
        """
        Add words to the list of stop words.

        Parameters
        ----------
        words : string or list of strings
            Words to remove from the stop words.

        Returns
        -------
        None.
        """
        if type(words) != list:
            words = [words]

        for word in words:
            # Prevent an error or trying to remove a word not in the list.
            if word not in cls.stopWords:
                cls.stopWords.append(word)


    @classmethod
    def ResetStopWordsList(cls):
        """
        Resets the stop words list.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        cls.stopWords = nltk.corpus.stopwords.words("english")


    @classmethod
    def RemoveStopWords(cls, text):
        result  = None

        if type(text) == pd.core.series.Series:
            result = text.apply(lambda entry : cls.RemoveStopWordsFromString(entry))
        elif type(text) == list:
            result = [token for token in text if token not in cls.stopWords]
        elif type(text) == str:
            result = cls.RemoveStopWordsFromString(text)

        return result


    @classmethod
    def RemoveStopWordsFromString(cls, text):
        result = text

        for word in cls.stopWords:
            pattern = r"(^|[^\w])" + word + r"\b"
            #pattern = r"(?:^|\W)rocket(?:$|\W)"
            result = re.sub(pattern, "", result)

        return result


    @classmethod
    def GetStopWords(cls, tokens):
        return [token for token in tokens if token in cls.stopWords]


    @classmethod
    def StripHtmlTags(cls, text):
        beautifulSoup = BeautifulSoup(text, "html.parser")
        return beautifulSoup.get_text()


    @classmethod
    def RemoveAccentedCharacters(cls, text):
        """
        Return the normal form form for the Unicode string.  Removes any accent characters.
        """
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
        return text


    @classmethod
    def Tokenize(cls, text):
        tokenizer = ToktokTokenizer()

        result  = None

        if type(text) == pd.core.series.Series:
            result = text.apply(lambda entry : tokenizer.tokenize(entry))
            result = result.apply(lambda tokens : [token.strip() for token in tokens])
        elif type(text) == list:
            result = []
            for i in range(len(text)):
                tokens = tokenizer.tokenize(text[i])
                result.append([token.strip() for token in tokens])
        elif type(text) == str:
            tokens = tokenizer.tokenize(text)
            result = [token.strip() for token in tokens]

        return result


    @classmethod
    def RemoveSpecialCharacters(cls, text, removeDigits=False):
        """
        Removes special characters.  This function automatically operates
        on the text in the correct way for different types of data structions.

        Parameters
        ----------
        text : Pandas DataFrame, list, or string
            The text to operate on.
        pattern : string
            regular expression to operate on.

        Returns
        -------
        text : Pandas DataFrame, list, or string
            The processed text.
        """
        pattern = r"[^a-zA-z\s]" if removeDigits else  r"[^a-zA-z0-9\s]"
        return LanguageHelper.ApplyRegularExpression(text, pattern)


    @classmethod
    def RemovePunctuation(cls, text):
        """
        Removes punctuation.  This function automatically operates
        on the text in the correct way for different types of data structions.

        Parameters
        ----------
        text : Pandas DataFrame, list, or string
            The text to operate on.
        pattern : string
            regular expression to operate on.

        Returns
        -------
        text : Pandas DataFrame, list, or string
            The processed text.
        """
        pattern = r"[^\w\s]"
        return LanguageHelper.ApplyRegularExpression(text, pattern)


    @classmethod
    def RemoveInternetHandles(cls, text):
        """
        Removes Twitter handles.  This function automatically operates
        on the text in the correct way for different types of data structions.

        Parameters
        ----------
        text : Pandas DataFrame, list, or string
            The text to operate on.
        pattern : string
            regular expression to operate on.

        Returns
        -------
        text : Pandas DataFrame, list, or string
            The processed text.
        """
        # @(\w{1,15})
        # Matches the @ followed by characters.
        # \b matches if the handle is followed by puncuation instead of space.
        # (^|[^@\w])
        # Removes extraneous spaces from around the match.
        pattern = r"(^|[^@\w])@(\w{1,15})\b"
        return LanguageHelper.ApplyRegularExpression(text, pattern)


    @classmethod
    def RemoveWebAddresses(cls, text):
        pattern = r"(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{0,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
        return LanguageHelper.ApplyRegularExpression(text, pattern)


    @classmethod
    def CreateWordCloud(cls, text, width=800, height=600, removeStopWords=False):
        text = LanguageHelper.RemoveInternetHandles(text)
        text = LanguageHelper.RemoveWebAddresses(text)
        text = LanguageHelper.ToLowerCase(text)

        if removeStopWords:
            text = LanguageHelper.RemoveStopWords(text)

        #text = LanguageHelper.RemovePunctuation(text)
        #text = LanguageHelper.RemoveSpecialCharacters(text)
        #text = LanguageHelper.StripHtmlTags(text)

        if type(text) != list:
            text = text.tolist()

        text = " ".join(text)

        wordcloud = WordCloud(
            stopwords=STOPWORDS,
            background_color="white",
            colormap="viridis",
            width=width,
            height=height,
            collocations=True
        ).generate(text)

        # Plot the wordcloud object.
        plt.imshow(wordcloud, interpolation="bilInear")
        plt.axis("off")
        plt.show()


    @classmethod
    def ApplyRegularExpression(cls, text, pattern, replaceString=""):
        """
        Applies a regular expression patttern to text.  This function automatically operates
        on the text in the correct way for different types of data structions.

        Parameters
        ----------
        text : Pandas DataFrame, list, or string
            The text to operate on.
        pattern : string
            regular expression to operate on.

        Returns
        -------
        text : Pandas DataFrame, list, or string
            The processed text.
        """
        result  = None

        if type(text) == pd.core.series.Series:
            result = text.apply(lambda entry : re.sub(pattern, replaceString, entry))
        elif type(text) == list:
            result = []
            for i in range(len(text)):
                result.append(re.sub(pattern, replaceString, text[i]))
        elif type(text) == str:
            result = re.sub(pattern, replaceString, text)

        return result


    @classmethod
    def ToLowerCase(cls, text):
        """
        Convert all characters to lowercase.  This function automatically operates
        on the text in the correct way for different types of data structions.

        Parameters
        ----------
        text : Pandas DataFrame, list, or string
            The text to operate on.
        pattern : string
            regular expression to operate on.

        Returns
        -------
        text : Pandas DataFrame, list, or string
            The processed text.
        """
        result = None

        if type(text) == pd.core.series.Series:
            result = text.apply(lambda entry : entry.lower())
        elif type(text) == list:
            result = []
            for i in range(len(text)):
                result.append(text[i].lower())
        elif type(text) == str:
            result = text.lower()

        return result


    @classmethod
    def ReplaceContractions(cls, text):
        """
        Replace contractions in string of text.
        """
        return contractions.fix(text)


    @classmethod
    def SimpleStemmer(cls, text):
        """
        Stemming using Porter Stemmer.
        """
        porterStemmer = nltk.porter.PorterStemmer()
        text          = " ".join([porterStemmer.stem(word) for word in text.split()])
        return text


    @classmethod
    def Lemmatize(cls, text):
        # !pip install spacy
        # !python -m spacy download en_core_web_sm
        # Install language packages using Anaconda environments.
        nlp  = spacy.load("en_core_web_sm")
        text = nlp(text)
        #text = " ".join([word.lemma_ if word.lemma_ != "-PRON-" else word.text for word in text])
        text = " ".join([word.text if word.lemma_ == "-PRON-" else word.lemma_ for word in text])
        return text