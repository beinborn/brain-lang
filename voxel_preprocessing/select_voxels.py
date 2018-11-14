import nilearn.signal
import scipy
import numpy as np

# METHODS TO REDUCE THE NUMBER OF VOXELS

# This method removes voxels with stdev 0 from the data.
# This is a necessary step before applying z-score.
# However, it should be applied reasonably.
# For example, it might not make much sense to first eliminate certain voxels and then do spatial smoothing or ROI selection.
def ignore_constant_voxels(data):
    print("Original shape: " + str(data.shape))
    selected_columns = np.where(np.std(data,0)==0)
    adjusted_data = np.delete(data, list(selected_columns), 1)
    print("After eliminating columns" + str(adjusted_data.shape))
    return adjusted_data

def select_voxels_by_variance(data):
    # TODO Samira, you already had a function for this, right?
    raise NotImplementedError()

def apply_PCA(data):
    # TODO add function for PCA
    raise NotImplementedError()

# TODO not yet tested
def region_of_interest_averaging(voxel_activations, voxel_to_region_mapping, regions_of_interest):
    roi_activation = []
    for roi in regions_of_interest:
        roi_voxels = []
        for voxel in voxel_activations:
            if voxel_to_region_mapping[voxel] in regions_of_interest:
                roi_voxels.append(voxel)
        roi_activation.append(np.average(roi_voxels))
