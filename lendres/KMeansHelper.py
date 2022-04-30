"""
Created on April 27, 2022
@author: Lance
"""
import numpy                          as np
from matplotlib                       import pyplot                     as plt
import matplotlib.cm                  as cm

from sklearn.cluster                  import KMeans
from scipy.stats                      import zscore
from scipy.spatial.distance           import cdist

from sklearn.metrics                  import silhouette_samples
from sklearn.metrics                  import silhouette_score

from lendres.ConsoleHelper            import ConsoleHelper
from lendres.PlotHelper               import PlotHelper


class KMeansHelper():

    def __init__(self, dataHelper):
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


    def CreateElbowPlot(self, data, clusters):
        """
        Constructor.

        Parameters
        ----------
        dataHelper : DataHelper
            DataHelper that has the data in a pandas.DataFrame.
        clusters : list of ints
            The clusters to calculate the distortions for.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The newly created figure.
        """
        meanDistortions = []

        for k in clusters:
            model = KMeans(n_clusters=k)
            model.fit_predict(data)

            meanDistortions.append(sum(np.min(cdist(data, model.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

        # Must be run before creating figure or plotting data.
        PlotHelper.FormatPlot()

        plt.plot(clusters, meanDistortions, "b-", marker="o", markerfacecolor="blue", markersize=12)
        plt.gca().set(title="Selecting the K Value (Number of Clusters) with the Elbow Method", xlabel="K Value", ylabel="Average Distortion")

        figure = plt.gcf()

        plt.show()

        return figure

    def CreateSilhouetteAnalysisPlots(self, data, rangeOfClusters):
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

        for clusters in rangeOfClusters:
            # Create a subplot with 1 row and 2 columns.
            title = "K-Means Clustering Silhouette Analysis with %d Clusters" % clusters
            figure, (leftAxis, rightAxis) = PlotHelper.NewSideBySideAxisFigure(title, width=15, height=6)

            # The 1st subplot is the silhouette plot.
            # The silhouette coefficient can range from -1, 1.
            #leftAxis.set_xlim([-1, 1])

            # The (clusters+1)*10 is for inserting blank space between silhouette.
            # Plots of individual clusters, to demarcate them clearly.
            leftAxis.set_ylim([0, len(X) + (clusters + 1) * 10])

            # Initialize the clusterer with clusters value and a random generator.
            # seed of 10 for reproducibility.
            clusterer     = KMeans(n_clusters=clusters, random_state=10)
            clusterLabels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed clusters.
            silhouetteAverage = silhouette_score(X, clusterLabels)
            print("For %d clusters" % clusters, "the average silhouette score is:", silhouetteAverage)

            # Compute the silhouette scores for each sample.
            sample_silhouette_values = silhouette_samples(X, clusterLabels)

            yLower = 10
            for i in range(clusters):
                # Aggregate the silhouette scores for samples belonging to cluster i, and sort them.
                ith_cluster_silhouette_values = sample_silhouette_values[clusterLabels == i]

                ith_cluster_silhouette_values.sort()

                numberOfSamplesInCluster = ith_cluster_silhouette_values.shape[0]
                yUpper = yLower + numberOfSamplesInCluster

                color = cm.nipy_spectral(float(i) / clusters)
                leftAxis.fill_betweenx(
                    np.arange(yLower, yUpper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7
                )

                # Label the silhouette plots with their cluster numbers at the middle.
                leftAxis.text(-0.05, yLower+0.5*numberOfSamplesInCluster, str(i), fontsize=12*PlotHelper.scale, verticalalignment="center")

                # Compute the new yLower for next plot.
                yLower = yUpper + 10

            # Set titles of left axis.
            leftAxis.set(xlabel="Silhouette Coefficient Values", ylabel="Cluster Label")
            PlotHelper.SetAxisToSquare(leftAxis)

            # The vertical line for average silhouette score of all the values.
            leftAxis.axvline(x=silhouetteAverage, color="red", linestyle="--")

            # Clear the yaxis labels / ticks.
            leftAxis.set_yticks([])
            #leftAxis.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

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