"""
Created on April 27, 2022
@author: Lance
"""
import numpy                          as np
from   matplotlib                     import pyplot                     as plt
import matplotlib.cm                  as cm
import matplotlib.ticker              as ticker
import seaborn                        as sns

from   sklearn.preprocessing          import StandardScaler
from   scipy.stats                    import zscore
from   scipy.spatial.distance         import cdist

from   sklearn.metrics                import silhouette_samples
from   sklearn.metrics                import silhouette_score

# To visualize the elbow curve and silhouette scores.
from   yellowbrick.cluster            import SilhouetteVisualizer

#from   lendres.ConsoleHelper          import ConsoleHelper
from   lendres.PlotHelper             import PlotHelper


class ClusterHelper():

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


    def FitPredict(self):
        self.model.fit_predict(self.scaledData)
        self.LabelData()


    def LabelData(self):
        self.dataHelper.data[self.labelColumn] = self.model.labels_


    def CreateBoxPlotForClusters(self):
        """
        Constructor.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        for column in self.columns:
            sns.boxplot(x=self.dataHelper.data[self.labelColumn], y=self.dataHelper.data[column])
            plt.show()