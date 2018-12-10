import nilearn.signal
import scipy
import numpy as np


# The methods here are used to preprocess the brain response.
# We preprocess train and test data separately although for steps like detrending,
# performance would probably be better when applied on the whole set.

def detrend(data, t_r, standardize=False):
    return nilearn.signal.clean(data, sessions=None,
                                detrend=True, standardize=standardize,
                                confounds=None, low_pass=None,
                                high_pass=0.005, t_r=t_r, ensure_finite=False)


# This method reduces the mean activation over all stimuli from each voxel.
# A related way of preprocessing is to subtract the activation of the resting scans from the stimuli scans.
def reduce_mean(data):
    return data - np.mean(data, axis=0)


# Wehbe et al, (2014) say that they apply a 5x5x5 Gaussian kernel.
# I assume that there is a setting for this in SPM.
# According to the answer here https://stackoverflow.com/questions/25216382/gaussian-filter-in-scipy
# it corresponds to the sigma and  truncate values as set below
# s = 3 w = 5, truncate = ((5-1)/2)-0.5) /2
# But I am not very confident about it.
# We did some testing and smooothing seemed to decrease the results.
def spatial_smoothing(voxel_activations, sigma=3, t=0.75):
    return scipy.ndimage.filters.gaussian_filter(voxel_activations, sigma, truncate=t)


# This method averages the voxel activation for all voxels in the list of regions of interest.
# We have not used it, but it should work.
def average_over_rois(voxel_activations, voxel_to_region_mapping, regions_of_interest):
    averaged_voxels = []
    for roi in regions_of_interest:
        # Get all voxels from region of interest
        voxel_indices = [value for key, value in voxel_to_region_mapping.items() if key == roi]

        roi_voxels = voxel_activations[voxel_indices]
        # Average over all voxels in roi
        averaged_voxels.extend(np.mean(roi_voxels))
    return averaged_voxels
