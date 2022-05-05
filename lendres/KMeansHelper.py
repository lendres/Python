"""
Created on April 27, 2022
@author: Lance
"""
import numpy                            as np
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

from   sklearn.cluster                  import AgglomerativeClustering
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


    def CreateElbowPlot(self, clusters):
        """
        Constructor.

        Parameters
        ----------
        clusters : list of ints
            The clusters to calculate the distortions for.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        """
        dataLength      = self.scaledData.shape[0]
        meanDistortions = []

        for k in clusters:
            self.model = KMeans(n_clusters=k, random_state=1)
            self.FitPredict()

            meanDistortions.append(sum(np.min(cdist(self.scaledData, self.model.cluster_centers_, "euclidean"), axis=1)) / dataLength)

        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        plt.plot(clusters, meanDistortions, "b-", marker="o", markerfacecolor="blue", markersize=12)
        plt.gca().set(title="Selecting the K Value (Number of Clusters) with the Elbow Method", xlabel="K Value", ylabel="Average Distortion")

        figure = plt.gcf()

        plt.show()

        return figure


    def CreateElbowPlot2(self, clusters):
        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        self.model = KMeans(random_state=1)
        visualizer = KElbowVisualizer(
            self.model,
            k=clusters,
            metric="silhouette",
            timings=False
        )

        visualizer.fit(self.scaledData)
        visualizer.show()


    def DisplaySilhouetteAnalysScores(self, rangeOfClusters):
        for clusters in rangeOfClusters:
            # Initialize the clusterer with clusters value and a random generator.
            # seed of 10 for reproducibility.
            #clusterer     = KMeans(n_clusters=clusters, random_state=1)
            clusterer = AgglomerativeClustering(n_clusters=clusters, affinity="euclidean", linkage="average")

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed clusters.
            clusterLabels = clusterer.fit_predict(self.scaledData)
            silhouetteAverage = silhouette_score(self.scaledData, clusterLabels)
            print("For %d clusters" % clusters, "the average silhouette score is:", silhouetteAverage)


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
        This is taken from:
            yellowbrick/yellowbrick/cluster/silhouette.py
        Prepare the figure for rendering by setting the title and adjusting the limits on the axes, adding labels and a legend.
        """
        # Set the title.
        if setTitle:
            if title == None:
                visualizer.set_title(("Silhouette Plot of {} Clustering with {} Centers").format(visualizer.name, visualizer.n_clusters_))
            else:
                visualizer.set_title(title)

        # Set the X and Y limits
        # The silhouette coefficient can range from -1, 1;
        # but here we scale the plot according to our visualizations

        # l_xlim and u_xlim are lower and upper limits of the x-axis,
        # set according to our calculated max and min score with necessary padding
        l_xlim = max(-1, min(-0.1, round(min(visualizer.silhouette_samples_) - 0.1, 1)))
        u_xlim = min(1, round(max(visualizer.silhouette_samples_) + 0.1, 1))
        visualizer.ax.set_xlim([l_xlim, u_xlim])

        # The (n_clusters_+1)*10 is for inserting blank space between
        # silhouette plots of individual clusters, to demarcate them clearly.
        visualizer.ax.set_ylim([0, visualizer.n_samples_ + (visualizer.n_clusters_ + 1) * 10])

        # Set the x and y labels
        visualizer.ax.set_xlabel("Silhouette Coefficient Values")
        visualizer.ax.set_ylabel("Cluster Label")

        # Set the ticks on the axis object.
        visualizer.ax.set_yticks(visualizer.y_tick_pos_)
        visualizer.ax.set_yticklabels(str(idx) for idx in range(visualizer.n_clusters_))
        # Set the ticks at multiples of 0.1
        visualizer.ax.xaxis.set_major_locator(ticker.MultipleLocator(xLabelIncrement))

        # Show legend (Average Silhouette Score axis)
        visualizer.ax.legend(loc="best")