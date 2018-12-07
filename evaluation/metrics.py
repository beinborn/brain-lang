from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
import numpy as np
import random
import scipy as sp
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from . import evaluation_util
import math
import logging
def mse(predictions, targets):
  """Mean Squared Error.
  :param predictions: (n_samples, n_outputs)
  :param targets: (n_samples, n_outputs)
  :return:
    a scalar which is mean squared error
  """
  return  mean_squared_error(predictions, targets)

def pairwise_matches(prediction1, target1, prediction2, target2):
    matches = {}
    for similarity_metric_fn in [cosine_similarity, euclidean_similarity, pearson_correlation]:
        correct1 = similarity_metric_fn(prediction1, target1)
        false1 = similarity_metric_fn(prediction1, target2)
        correct2 = similarity_metric_fn(prediction2, target2)
        false2 = similarity_metric_fn(prediction2, target1)

        metric_name = similarity_metric_fn.__name__
        matches[metric_name +"_Mitchell"] = int((correct1 + correct2) > (false1 + false2))
        matches[metric_name + "_Wehbe1"] = int(correct1 > false1)
        matches[metric_name + "_Wehbe2"] = int(correct2 > false2)
        matches[metric_name + "_Strict"] = int((correct1 >false1) & (correct2 > false2))

    return matches


# Choose a random correct prediction/target pair and select a random
# incorrect prediction for comparison
def pairwise_accuracy_randomized(predictions, targets, number_of_trials):
    # We do not want to compare directly neighbouring stimuli because of the hemodynamic response pattern
    # The random sample should thus be at least 20 steps ahead
    constraint = 20
    collected_results = {}
    for trial in range(0, number_of_trials):
        for i in range(0, len(predictions)):
            prediction1 = predictions[i]
            target1 = targets[i]
            index_for_pair = random.randint(0, len(predictions)-1)
            # Get a random value that does not fall within the constrained region
            while abs(i - index_for_pair) < constraint:
                index_for_pair = random.randint(0, len(predictions)-1)

            prediction2 = predictions[index_for_pair]
            target2 = targets[index_for_pair]
            matches = pairwise_matches(prediction1, target1, prediction2, target2)
            collected_results = evaluation_util.add_to_collected_results(matches, collected_results)

    averaged_results = {}
    for key, matches in collected_results.items():
        avg_trial_matches = matches / float(number_of_trials)
        averaged_results[key] = avg_trial_matches

    return averaged_results





### EVALUATION METHODS ###

def cosine_similarity(vector1, vector2):

    return 1 - sp.spatial.distance.cosine(vector1, vector2)

def euclidean_similarity(vector1, vector2):
    return 1 - sp.spatial.distance.euclidean(vector1, vector2)

def pearson_correlation(vector1,vector2):
    return pearsonr(vector1, vector2)[0]


def r2_score_complex(predictions, targets):
    r2values = r2_score( targets,predictions, multioutput="raw_values")
    return r2values, np.mean(np.asarray(r2values)), np.sum(np.asarray(r2values))

def explained_variance_complex(predictions, targets):
    ev_scores = explained_variance_score( targets,predictions, multioutput="raw_values")
    return ev_scores, np.mean(np.asarray(ev_scores)), np.sum(np.asarray(ev_scores))

def explained_variance(predictions, targets):
    return explained_variance_score( targets,predictions, multioutput="raw_values")
def pearson_complex(predictions, targets):
    correlations_per_voxel = []
    for voxel_id in range(0,len(targets[0])):
        correlation = pearsonr(predictions[:,voxel_id],  targets[:, voxel_id])[0]
        correlations_per_voxel.append(correlation)
    return np.asarray(correlations_per_voxel), np.mean(np.asarray(correlations_per_voxel)), np.sum(np.asarray(correlations_per_voxel))

def pearson_jain_complex(predictions, targets):
    corr_squared_per_voxel = []
    for voxel_id in range(0,len(targets[0])):
        correlation = pearsonr(predictions[:,voxel_id],  targets[:, voxel_id])[0]
        if(math.isnan(correlation)):
            print("Encountered NaN value")
            print("Voxel: " + str(voxel_id))
            print("Predictions")
            print(predictions[:,voxel_id])
            print("Targets")
            print(targets[:,voxel_id])
        corr_squared = abs(correlation) * correlation
        print("Correlation: " + str(correlation))
        print("Abs(correlation) * correlation: " + str(corr_squared))
        corr_squared_per_voxel.append(corr_squared)
    return np.asarray(corr_squared_per_voxel), np.mean(np.asarray(corr_squared_per_voxel)), np.sum(np.asarray(corr_squared_per_voxel))




# Methods for calculating dissimilarity matrices
def get_dists(data):
    logging.info("Calculating dissimilarity matrices")
    x = {}
    C = {}

    for i in np.arange(len(data)):
        x[i] = data[i]
        C[i] = sp.spatial.distance.cdist(x[i], x[i], 'cosine') + 0.00000000001
        C[i] /= C[i].max()

    return x, C

# Calculate correlation between
def compute_distance_over_dists(x, C):
    logging.info("Calculate ")
    keys = np.asarray(list(x.keys()))
    correlations = np.zeros((len(keys), len(keys)))
    for i in np.arange(len(keys)):
        for j in np.arange(len(keys)):
            corr = []
            for a, b in zip(C[keys[i]], C[keys[j]]):
                p, _ = sp.stats.spearmanr(a, b)
                corr.append(p)

        correlations[i][j] = np.mean(corr)

    return correlations



