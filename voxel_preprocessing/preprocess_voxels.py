import nilearn.signal
import scipy
import numpy as np

def detrend(timeseries_datapoints, t_r, standardize=False):
    return nilearn.signal.clean(timeseries_datapoints, sessions=None,
                         detrend=True, standardize=standardize,
                         confounds=None, low_pass=None,
                         high_pass=0.005, t_r=t_r, ensure_finite=False)


# Transform voxel values into z-scores. (x-mean)/stdev
# This only works if stdev is NOT 0. This is the case for constant voxels.
def zscore(data):
    return scipy.stats.zscore(data)

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

def minus_average_resting_states(timeseries_datapoints, brain_states_with_no_stimuli):
  """
  :param timeseries_datapoints:
  :param brain_states_with_no_stimuli:
  :return:
  """

  # For now we simply normalize by minusing the avg resting state.
  average_brain_state_with_no_stimuli = np.mean(brain_states_with_no_stimuli, axis=-1)
  timeseries_datapoints = timeseries_datapoints - average_brain_state_with_no_stimuli

  return timeseries_datapoints

# Wehbe et al apply a 5x5x5 Gaussian kernel.
# According to the answer here https://stackoverflow.com/questions/25216382/gaussian-filter-in-scipy
# This corresponds to the sigma and  truncate values as set below
# s = 3 w = 5, truncate = ((5-1)/2)-0.5) /2
# But I am VERY unsure about this!
# TODO if we use it, need to better understand this
def spatial_smoothing(voxel_activations, sigma = 3, truncate = 0.75 ):
    return scipy.ndimage.filters.gaussian_filter(voxel_activations, sigma, )

# TODO not yet tested
def region_of_interest_averaging(voxel_activations, voxel_to_region_mapping, regions_of_interest):
    roi_activation = []
    for roi in regions_of_interest:
        roi_voxels = []
        for voxel in voxel_activations:
            if voxel_to_region_mapping[voxel] in regions_of_interest:
                roi_voxels.append(voxel)
        roi_activation.append(np.average(roi_voxels))
