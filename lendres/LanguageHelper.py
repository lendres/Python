"""
Created on July 16, 2022
@author: Lance A. Endres
"""

import unicodedata
from   bs4                                       import BeautifulSoup

# Natural language processing tool-kit
import nltk
from   nltk.tokenize.toktok                      import ToktokTokenizer
import contractions

import re
import spacy

from   IPython.display                           import display

import lendres
from   lendres.ConsoleHelper                     import ConsoleHelper


class LanguageHelper():
    stopWords = nltk.corpus.stopwords.words("english")


    @classmethod
    def Install(cls):
       nltk.download("stopwords")


    @classmethod
    def RemoveStopWords(cls, words):
        if type(words) != list:
            words = [words]

        for word in words:
            # Prevent an error or trying to remove a word not in the list.
            if word in cls.stopWords:
                cls.stopWords.remove(word)


    @classmethod
    def FilterOutStopWords(cls, tokens):
        return [token for token in tokens if token not in cls.stopWords]


    @classmethod
    def StripHtmlTags(cls, text):
        beautifulSoup = BeautifulSoup(text, "html.parser")
        return beautifulSoup.get_text()


    @classmethod
    def RemoveAccentedCharacters(cls, text):
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
        return text


    @classmethod
    def Tokenize(cls, text):
        tokenizer = ToktokTokenizer()
        tokens    = tokenizer.tokenize(text)
        return [token.strip() for token in tokens]


    @classmethod
    def RemoveSpecialCharacters(cls, text, removeDigits=False):
        pattern = r"[^a-zA-z\s]" if removeDigits else  r"[^a-zA-z0-9\s]"
        return re.sub(pattern, "", text)


    def RemovePunctuation(cls, text):
        return re.sub(r"[^\w\s]", "", text)


    @classmethod
    def SimpleStemmer(cls, text):
        porterStemmer = nltk.porter.PorterStemmer()
        text          = " ".join([porterStemmer.stem(word) for word in text.split()])
        return text


    @classmethod
    def ToLowerCase(cls, words):
        """
        Convert all characters to lowercase.
        If the input is a list, the operation is performed in place.
        """
        if type(words) == list:
            for i in range(len(words)):
                words[i] = words[i].lower()
        else:
            words = words.lower()
        return words


    @classmethod
    def ReplaceContractions(cls, text):
        """
        Replace contractions in string of text.
        """
        return contractions.fix(text)


    @classmethod
    def Lemmatize(cls, text):
        # !pip install spacy
        # !python -m spacy download en_core_web_sm
        # Install language packages using Anaconda environments.
        nlp  = spacy.load("en_core_web_sm")
        text = nlp(text)
        text = " ".join([word.lemma_ if word.lemma_ != "-PRON-" else word.text for word in text])
        #text = " ".join([word.text if word.lemma_ == "-PRON-" else word.lemma_ for word in text])
        return text