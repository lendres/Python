"""
Created on July 27, 2022
@author: Lance A. Endres
"""

from   sklearn.feature_extraction.text           import CountVectorizer
from   sklearn.feature_extraction.text           import TfidfVectorizer

from   lendres.DataHelper                        import DataHelper


class LanguageDataHelper(DataHelper):


    def __init__(self, fileName=None, data=None, copy=False, deep=False, consoleHelper=None):
        """
        Constructor.

        Parameters
        ----------
        fileName : stirng, optional
            Path to load the data from.  This is a shortcut for creating a DataHelper and
            then calling "LoadAndInspectData."
        data : pandas.DataFrame, optional
            DataFrame to operate on. The default is None.  If None is specified, the
            data should be loaded in a separate function call, e.g., with "LoadAndInspectData"
            or by providing a fileName to load the data from.  You cannot provide both a file
            and data.
        deep : bool, optional
            Specifies if a deep copy should be done. The default is False.  Only valid if
            the "data" parameter is specified.
        consoleHelper : ConsoleHelper
            Class the prints messages.

        Returns
        -------
        None.
        """
        super().__init__(fileName, data, copy, deep, consoleHelper)

        self.vectorizer = None


    def Vectorize(self, sourceColumn, method="tfidf", **kwargs):
        """
        Vectorizes the data based on the specified method.
        Note, the data MUST be split first.

        Parameters
        ----------
        sourceColumn : string
            Column in self.data that the source material (used for the "fit") is found in.
        method : stirng, optional
            Method used to vectorize the text.
        **kwargs : keyword arguments
            These arguments are passed on to the vectorizer.

        Returns
        -------
        None.
        """
        if len(self.xTrainingData) == 0:
            raise Exception("The data has not been split.")

        if method == "count":
            self.vectorizer = CountVectorizer(**kwargs)
        elif method == "tfidf":
            self.vectorizer = TfidfVectorizer(**kwargs)
        else:
            raise Exception("Invalid vectorization method specified.")

        # Fit the model based on the source material.
        self.vectorizer.fit(self.data[sourceColumn])

        # Now transform the data.
        self.xTrainingData = self.vectorizer.transform(self.xTrainingData[sourceColumn]).toarray()
        self.xTestingData  = self.vectorizer.transform(self.xTestingData[sourceColumn]).toarray()

        if len(self.xValidationData) != 0:
            self.xValidationData = self.vectorizer.transform(self.xValidationData[sourceColumn]).toarray()