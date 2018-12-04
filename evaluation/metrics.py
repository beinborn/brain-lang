from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
import numpy as np
import random
import scipy.spatial as sp
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from . import evaluation_util


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

def first_order_rdm(data, distance_metric):
    RDMs = []
    for i in range(len(data)):
      RDM = sp.distance.cdist(np.asarray(data[i]),np.asarray(data[i]), distance_metric)
      RDMs.append(RDM)
    return RDMs

def second_order_rdm(RDMs):
    flat = [m.flatten(1) for m in RDMs]
    flat = np.array(flat).transpose()
    c_matrix = spearmanr(flat)[0]
    if not(isinstance(c_matrix, np.ndarray)):
        c_matrix = np.array([[1,c_matrix],[c_matrix,1]])
    RDM = np.ones(c_matrix.shape) - c_matrix
    return RDM




### EVALUATION METHODS ###

def cosine_similarity(vector1, vector2):

    return 1 - sp.distance.cosine(vector1, vector2)

def euclidean_similarity(vector1, vector2):
    return 1 - sp.distance.euclidean(vector1, vector2)

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




# TODO not yet working, debug
def representational_similarity_analysis(scans, embeddings, distance_metric='cosine'):
    # If we do this, we do not need to apply a mapping model
    # Calculate matrix of distances between all pairs of scans
    # Calculate matrix of distances between all pairs of embeddings
    l = scans + embeddings
    RDMS = first_order_rdm(l, distance_metric)
    # for all possible pairs of (scan, embedding) calculate
    # correlation between distance vectors in similarity metrics
    RDM = second_order_rdm(RDMS)







