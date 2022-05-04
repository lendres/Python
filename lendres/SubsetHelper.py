"""
Created on April 27, 2022
@author: Lance
"""
from   sklearn.preprocessing                     import StandardScaler
from   scipy.stats                               import zscore



class SubsetHelper():

    def __init__(self, dataHelper, columns, copyMethod="include"):
        """
        Constructor.

        Parameters
        ----------
        dataHelper : DataHelper
            DataHelper that has the data in a pandas.DataFrame.

        Returns
        -------
        None.
        """
        self.dataHelper                = dataHelper
        self.model                     = None
        self.labelColumn               = "Cluster Label"
        self.columns                   = None
        self.scaledData                = None

        if copyMethod == "include":
            self.columns = columns
        elif copyMethod == "exclude":
            self.columns = self.dataHelper.data.columns.values.tolist()
            for column in columns:
                self.columns.remove(column)
        else:
            raise Exception("The copy method specified is invalid.")

        self.scaledData = self.dataHelper.data[self.columns].copy(deep=True)


    def ScaleData(self, method="standardscaler"):
        """
        Constructor.

        Parameters
        ----------
        method : string
            Method used to normalized the data.
            standardscaler : Uses the StandardScaler class.
            zscore : Uses the zscore.

        Returns
        -------
        None.
        """
        if method == "standardscaler":
            scaler                  = StandardScaler()
            self.scaledData         = scaler.fit_transform(self.scaledData)
        elif method == "zscore":
            self.scaledData         = self.scaledData.apply(zscore).to_numpy()
        else:
            raise Exception("The specified scaling method is invalid.")