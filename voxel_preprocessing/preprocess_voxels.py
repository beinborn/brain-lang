import nilearn.signal
import scipy
import numpy as np

### METHODS APPLIED OVER ALL VOXELS

def detrend(data, t_r, standardize=False):
    return nilearn.signal.clean(data, sessions=None,
                         detrend=True, standardize=standardize,
                         confounds=None, low_pass=None,
                         high_pass=0.005, t_r=t_r, ensure_finite=False)


def reduce_mean(data):
  return data - np.mean(data, axis=0)



# Wehbe et al apply a 5x5x5 Gaussian kernel.
# According to the answer here https://stackoverflow.com/questions/25216382/gaussian-filter-in-scipy
# This corresponds to the sigma and  truncate values as set below
# s = 3 w = 5, truncate = ((5-1)/2)-0.5) /2
# But I am VERY unsure about this!
# TODO if we use it, need to better understand this
def spatial_smoothing(voxel_activations, sigma = 3, t = 0.75 ):
    return scipy.ndimage.filters.gaussian_filter(voxel_activations, sigma, truncate = t)

# TODO not yet tested
def average_over_rois(voxel_activations, voxel_to_region_mapping, regions_of_interest):
    averaged_voxels = []
    for roi in regions_of_interest:
        # Get all voxels from region of interest
        voxel_indices = [value for key, value in voxel_to_region_mapping.items() if key == roi]
        roi_voxels = voxel_activations[voxel_indices]
        # average over all voxels in roi
        averaged_voxels.extend(np.mean(roi_voxels))
    return averaged_voxels