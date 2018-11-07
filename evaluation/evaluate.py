# In: List of voxel vectors, list of stimuli vectors
# Out: Score for quality of stimuli vectors
# Challenge: How can we best evaluate representations for continuous language?

# Parameters:
# evaluation measure: 2x2, accuracy, explained variance, ...?
# setup: cross-validation setting, train-test split etc
# hemodynamic delay
# mode of representation combination (concatenation, average etc)
# --> maybe the combination mode could also be decided earlier when loading representations

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import explained_variance_score
from sklearn import preprocessing
import numpy as np

# Evaluate if the correlation between the predicted representation and the gold data is higher
# than the correlation between the predicted representation and a random data sample
# This is the evaluation method used by Mitchell et al. (2008)
def pairwise_cosine(gold_data, prediction, random_data):
    similarity_with_prediction = cosine_similarity(prediction, gold_data)
    random_similarity = cosine_similarity(prediction, random_data)
    return similarity_with_prediction >= random_similarity

def pairwise_euclidean(scan, representation, random_scan):
    distance_to_true_scan = np.linalg.norm(preprocessing.normalize(representation) - preprocessing.normalize(scan))
    distance_to_random_scan = np.linalg.norm(preprocessing.normalize(representation) - preprocessing.normalize(random_scan))
    return distance_to_true_scan <= distance_to_random_scan

# Gold_data and predictions are matrices of the same size with the following structure
# Rows correspond to voxels, columns correspond to time steps
# We get the prediction matrix by feeding the stimulus to a model that has learned a linear mapping from stimulus to brain activation.
# TODO: this method is redundant here, because it does not do anything that sklearn does not already do!
def voxelwise_explainedVariance(gold_data, predictions):
    return explained_variance_score(gold_data, predictions)
