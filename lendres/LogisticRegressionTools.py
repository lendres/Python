"""
Created on January 19, 2022
@author: Lance A. Endres
"""
import pandas                               as pd
import numpy                                as np

from sklearn                                import metrics

class LogisticRegressionTools():


    @classmethod
    def GetRocCurveAndScores(cls, y, yPredictedProbabilities):
        """
        Calculates the area under the curve and geomtric mean.

        Parameters
        ----------
        y : array like of ints
            True values.
        yPredicted : array like of doubles
            Predicted probabilities.
        Returns
        -------
        falsePositiveRates : array of doubles
            False positive rates (range 0-1).
        truePositiveRates : array of doubles
            True positive rates (range 0-1).
        scores : pandas.DataFrame
            DataFrame in the format:
                area_under_curve   best_geometric_mean    best_threshold   best_threshold_index
        """
        # Get the area under the curve score.
        aucScore                                          = metrics.roc_auc_score(y, yPredictedProbabilities)
        falsePositiveRates, truePositiveRates, thresholds = metrics.roc_curve(y, yPredictedProbabilities)

        # Calculate the geometric mean for each threshold.
        #
        # The true positive rate is called the Sensitivity. The inverse of the false-positive rate is called the Specificity.
        # Sensitivity = True Positive / (True Positive + False Negative)
        # Specificity = True Negative / (False Positive + True Negative)
        # Where:
        #     Sensitivity = True Positive Rate
        #     Specificity = 1 – False Positive Rate
        #
        # The geometric mean is a metric for imbalanced classification that, if optimized, will seek a balance between the
        # sensitivity and the specificity.

        # geometric mean = sqrt(Sensitivity * Specificity)
        geometricMeans = np.sqrt(truePositiveRates * (1-falsePositiveRates))

        # Locate the index of the largest geometric mean.
        index = np.argmax(geometricMeans)

        scores  = pd.DataFrame(
            [[aucScore, geometricMeans[index], thresholds[index], index]],
            index=["Best"],
            columns=["Area Under Curve", "Best Geometric Mean", "Best Threshold", "Index of Best Threshold"]
        )

        return falsePositiveRates, truePositiveRates, scores