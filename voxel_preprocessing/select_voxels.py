import nilearn.signal

from evaluation.metrics import *
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score

# ----- Methods to reduce the number of voxels -----

# This method returns all voxel ids of voxels with stdev 0 in the data.
# This is like removing the constant voxels, but we want to make sure that the ids remain the same.
# Removing constant voxels is a necessary step before applying z-score.
# However, it should be applied reasonably.
# For example, it might not make much sense to first eliminate certain voxels and then do spatial smoothing or ROI selection.
# Note that constant voxels mess up the evaluation results because their explained variance will be 1.

def select_varied_voxels(train_activations):
    selected_ids = np.where(abs(np.std(train_activations, 0)) > 0)[0]
    return selected_ids.tolist()


# Make sure that model-driven voxel selection is ONLY performed on the train_sets, or even better on a development set.
def select_voxels_by_variance(train_predictions, train_targets):

    # calculate explained variance
    ev_per_voxel = explained_variance_score( train_targets,train_predictions, multioutput="raw_values")
    # get ids for  n most predictive voxels
    sorted_voxels_ids = sorted(range(len(ev_per_voxel)), key = lambda i : ev_per_voxel[i])
    print("Retrieved ev values: " + str(len(ev_per_voxel)))
    print("Sorted voxel ids: " + str(len(sorted_voxels_ids)))
    # return list of ids of voxels
    return sorted_voxels_ids

def select_voxels_by_r(train_predictions, train_targets):
    # Make sure that this selection is ONLY performed on the train_sets!
    # We calculate r as in Jain & Huth, 2018

    correlations = pearson_complex(train_predictions, train_targets)[0]
    # get ids for  n most predictive voxels
    sorted_voxels_ids = sorted(range(len(correlations)), key = lambda i : correlations[i])
    print("Retrieved r values: " + str(len(correlations)))
    print("Sorted voxel ids: " + str(len(sorted_voxels_ids)))
    return sorted_voxels_ids

    # return list of ids of voxels
    return topn_voxels_ids

# Theory-driven voxel selection:
# Determine a list of regions of interest and only analyze voxels in these regions.
def select_voxels_by_roi(voxel_activations, voxel_to_region_mapping, regions_of_interest):
    roi_voxels = []

    for index in range(0, len(voxel_activations)):

        if voxel_to_region_mapping[index] in regions_of_interest:
            roi_voxels.append(index)

    return roi_voxels
