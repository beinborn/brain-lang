import nilearn.signal
import scipy
import numpy as np

from evaluation.metrics import explained_variance
# METHODS TO REDUCE THE NUMBER OF VOXELS

# This method removes voxels with stdev 0 from the data.
# This is a necessary step before applying z-score.
# However, it should be applied reasonably.
# For example, it might not make much sense to first eliminate certain voxels and then do spatial smoothing or ROI selection.
def ignore_constant_voxels(train_activations):
    selected_ids = np.where(abs(np.std(train_activations, 0)) > 0)[0]
    return selected_ids.tolist()


#TODO not yet tested
def select_voxels_by_variance(train_predictions, train_targets, n):
    # TODO Make sure that this selection is ONLY performed on the train_sets
    # calculate explained variance
    ev_per_voxel = explained_variance(train_predictions, train_targets)
    # get ids for  n most predictive voxels
    topn_voxels_ids = sorted(range(len(ev_per_voxel)), key = lambda i : ev_per_voxel[i])[-n:]
    print(topn_voxels_ids)
    # return list of ids of voxels
    return topn_voxels_ids


# TODO not yet tested
def select_voxels_by_roi(voxel_activations, voxel_to_region_mapping, regions_of_interest):
    roi_voxels = []

    for index in range(0, len(voxel_activations)):

        if voxel_to_region_mapping[index] in regions_of_interest:
            roi_voxels.append(index)
            print(index)
    print(roi_voxels)
    return roi_voxels
