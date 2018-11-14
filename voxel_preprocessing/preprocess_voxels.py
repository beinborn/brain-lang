import nilearn.signal
import scipy
import numpy as np

### METHODS APPLIED OVER ALL VOXELS

def detrend(timeseries_datapoints, t_r, standardize=False):
    return nilearn.signal.clean(timeseries_datapoints, sessions=None,
                         detrend=True, standardize=standardize,
                         confounds=None, low_pass=None,
                         high_pass=0.005, t_r=t_r, ensure_finite=False)


# Transform voxel values into z-scores. (x-mean)/stdev
# This only works if stdev is NOT 0. This is the case for constant voxels.
def zscore(data):
    zscores = scipy.stats.zscore(data)
    if (np.isnan(zscores).any())
        raise ValueError("Data contains voxels with stdev 0")
    else:
        return zscores



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
def spatial_smoothing(voxel_activations, sigma = 3, t = 0.75 ):
    return scipy.ndimage.filters.gaussian_filter(voxel_activations, sigma, truncate = t)
