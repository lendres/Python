"""
Created on April 27, 2022
@author: Lance A. Endres
"""
import pandas                           as pd
import numpy                            as np
import matplotlib
from   matplotlib                       import pyplot                     as plt
import matplotlib.cm                    as cm
import matplotlib.ticker                as ticker

from   sklearn.cluster                  import KMeans
from   scipy.spatial.distance           import cdist

from   sklearn.metrics                  import silhouette_samples
from   sklearn.metrics                  import silhouette_score

# To visualize the elbow curve and silhouette scores.
from   yellowbrick.cluster              import KElbowVisualizer
from   yellowbrick.cluster              import SilhouetteVisualizer

#from   lendres.ConsoleHelper            import ConsoleHelper
from   lendres.PlotHelper               import PlotHelper
from   lendres.ClusterHelper            import ClusterHelper

class KMeansHelper(ClusterHelper):

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
        super().__init__(dataHelper, columns, copyMethod)


    def CreateModel(self, clusters):
        self.model = KMeans(n_clusters=clusters, random_state=1)


    def CreateVisualizerPlot(self, clusters, metric="silhouette"):
        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        self.model = KMeans(random_state=1)
        visualizer = KElbowVisualizer(
            self.model,
            k=clusters,
            metric=metric,
            timings=False
        )

        # Fix marker size.  Needs to be done before the call to "fit."
        plt.rcParams.update({"lines.markersize" : 8*PlotHelper.scale})

        # Creates the plot.
        visualizer.fit(self.scaledData)

        # Final formating.
        visualizer.ax.set_title(metric.title() + " Score for K Means Clustering")
        visualizer.ax.set_xlabel("K Value")
        visualizer.ax.set_ylabel(metric.title() + " Score")
        visualizer.ax.legend(loc="best", frameon=True)

        # Figure must be saved before calling show.
        figure = plt.gcf()

        # Don't call visualizer.show() or it will overwrite the formatting.
        plt.show()

        return figure


    def GetSilhouetteAnalysScores(self, rangeOfClusters):
        """
        Gets a DataFrame of the average silhouette score for the range of clusters.

        Parameters
        ----------
        rangeOfClusters : Range
            Range of clusters to get the silhouette scores for.

        Returns
        -------
        comparisonFrame : pandas.Dataframe
            A DataFrame that has the average silhouette scores.
        """
        columnsLabels   = ["Clusters", "Average Silhouette Score"]
        comparisonFrame = pd.DataFrame(columns=columnsLabels)

        for clusters in rangeOfClusters:
            # Initialize the clusterer with clusters value and a random generator.
            # seed of 10 for reproducibility.
            clusterer         = KMeans(n_clusters=clusters, random_state=1)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed clusters.
            clusterLabels     = clusterer.fit_predict(self.scaledData)
            silhouetteAverage = silhouette_score(self.scaledData, clusterLabels)

            row = pd.DataFrame([[clusters, silhouetteAverage]], columns=columnsLabels)
            comparisonFrame   = pd.concat([comparisonFrame, row], ignore_index=True)

        comparisonFrame["Clusters"] = comparisonFrame["Clusters"].astype("int32")
        comparisonFrame.set_index("Clusters", inplace=True)
        return comparisonFrame


    def CreateSilhouetteAnalysisPlots(self, rangeOfClusters):
        """
        Constructor.

        Parameters
        ----------
        dataHelper : DataHelper
            DataHelper that has the data in a pandas.DataFrame.
        rangeOfClusters : list of ints
            The range of clusters to calculate the distortions for.

        Returns
        -------
        None.
        """
        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        for clusters in rangeOfClusters:

            visualizer = SilhouetteVisualizer(KMeans(n_clusters=clusters, random_state=1))
            visualizer.fit(self.scaledData)

            # Finalize on the visualizer to do most of the formatting.  Then call our version
            # of finalize to make selected formatting changes.  Finally, call show.  We cannot
            # call the model.show() version because that will call plt.show() and the axis and
            # figure will no longer be available for formatting.
            visualizer.finalize()
            self.SihlouettePlotFinalize(visualizer)
            plt.show()


    def CreateTwoColumnSilhouetteVisualizationPlots(self, data, rangeOfClusters):
        """
        Constructor.

        Parameters
        ----------
        dataHelper : DataHelper
            DataHelper that has the data in a pandas.DataFrame.
        rangeOfClusters : list of ints
            The range of clusters to calculate the distortions for.

        Returns
        -------
        None.
        """
        X = data

        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        for clusters in rangeOfClusters:

            # Create a subplot with 1 row and 2 columns.
            title = "K-Means Clustering Silhouette Analysis with %d Clusters" % clusters
            figure, (leftAxis, rightAxis) = PlotHelper.NewSideBySideAxisFigure(title, width=15, height=6)

            # Initialize the clusterer with clusters value and a random generator.
            # seed of 10 for reproducibility.
            clusterer     = KMeans(n_clusters=clusters, random_state=1)

            colors = cm.nipy_spectral(np.arange(0, clusters) / float(clusters))
            visualizer = SilhouetteVisualizer(KMeans(n_clusters=clusters, random_state=1), ax=leftAxis, colors=colors)
            visualizer.fit(X)

            # Axis must be set to square after call finalize.
            visualizer.finalize()
            self.SihlouettePlotFinalize(visualizer, setTitle=False, xLabelIncrement=0.2)
            PlotHelper.SetAxisToSquare(leftAxis)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed clusters.
            clusterLabels = clusterer.fit_predict(X)

            # Right plot showing the actual clusters formed.
            colors = cm.nipy_spectral(clusterLabels.astype(float) / clusters)
            rightAxis.scatter(X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

            # Labeling the clusters.
            centers = clusterer.cluster_centers_

            # Draw white circles at cluster centers.
            rightAxis.scatter(centers[:, 0], centers[:, 1], marker="o", c="white", alpha=1, s=200, edgecolor="k")

            # Label circles.
            for i, c in enumerate(centers):
                rightAxis.text(c[0], c[1], str(i), fontsize=12*PlotHelper.scale, horizontalalignment="center", verticalalignment="center")
                #rightAxis.scatter(c[0]+1, c[1]+1, marker="$%d$" % i, alpha=1, s=100, edgecolor="k")

            rightAxis.set(xlabel="Feature Space for the 1st Feature", ylabel="Feature Space for the 2nd Feature")
            PlotHelper.SetAxisToSquare(rightAxis)

            plt.show()


    def SihlouettePlotFinalize(self, visualizer, setTitle=True, title=None, xLabelIncrement=0.1):
        """
        Prepare the figure for rendering by setting the title and adjusting the limits on the axes, adding labels and a legend.
        """
        # Set the title.
        if setTitle:
            if title == None:
                visualizer.set_title(("Silhouette Plot of {} Clustering with {} Centers").format(visualizer.name, visualizer.n_clusters_))
            else:
                visualizer.set_title(title)
        else:
            visualizer.set_title("")

        # Set the x and y labels
        visualizer.ax.set_xlabel("Silhouette Coefficient Values")
        visualizer.ax.set_ylabel("Cluster Label")

        # Set the ticks multiples.
        visualizer.ax.xaxis.set_major_locator(ticker.MultipleLocator(xLabelIncrement))