import nilearn.signal
from scipy import stats
import numpy as np

def detrend(timeseries_datapoints, t_r, standardize=False):
    return nilearn.signal.clean(timeseries_datapoints, sessions=None,
                         detrend=True, standardize=standardize,
                         confounds=None, low_pass=None,
                         high_pass=0.005, t_r=t_r, ensure_finite=False)


# Transform voxel values into z-scores. (x-mean)/stdev
# This only works if stdev is NOT 0. This is the case for constant voxels.
def zscore(data):
    return stats.zscore(data)

# This method removes voxels with stdev 0 from the data.
# This is a necessary step before applying z-score.
# However, it should be applied reasonably.
# For example, it might not make much sense to first eliminate certain voxels and then do spatial smoothing or ROI selection.
def select(data):
    print("Original shape: " + str(data.shape))
    selected_columns = np.where(np.std(data,0)==0)
    adjusted_data = np.delete(data, list(selected_columns), 1)
    print("After eliminating columns" + str(adjusted_data.shape))
    return adjusted_data


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