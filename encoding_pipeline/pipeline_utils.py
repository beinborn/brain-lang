import numpy as np
import logging


# Reduce the voxels to the set of interesting voxel ids
def reduce_voxels(list_of_scans, interesting_voxel_ids):

    if len(interesting_voxel_ids) < len(list_of_scans[0]):
        reduced_list = np.asarray([list(scan[interesting_voxel_ids]) for scan in list_of_scans])
    else:
        return np.asarray(list_of_scans)
    return reduced_list


def evaluate_fold(metrics, predictions, targets):
    logging.info("Evaluating...")
    results = {}

    for metric_name, metric_fn in metrics.items():
        results[metric_name] = metric_fn(predictions, targets)
    return results


def preprocess_voxels(voxel_preprocessings, scans):
    # Important note: Order of preprocessing matters!
    for voxel_preprocessing_fn, args in voxel_preprocessings:
        scans = voxel_preprocessing_fn(scans, **args)
    return scans
