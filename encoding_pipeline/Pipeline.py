"""Pipeline for predicting brain activations.
"""

from evaluation.metrics import *

from voxel_preprocessing.select_voxels import *
from read_dataset.read_words_data import MitchellReader
import numpy as np

import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)


class Pipeline(object):
    def __init__(self, brain_data_reader, stimuli_encoder, mapper, save_dir="processed_data/"):
        self.brain_data_reader = brain_data_reader
        self.stimuli_encoder = stimuli_encoder
        self.mapper = mapper
        self.subject_ids = [1]
        self.data = {}
        self.voxel_preprocessings = []
        # Options for voxel selection are: "on_train","random", "by_roi", and "none"
        self.voxel_selection = "on_train"
        # Example: [(detrend, {'t_r': 2.0})]
        self.voxel_selections = []
        self.roi = []
        self.save_dir = save_dir
        self.load_previous = False
        self.voxel_to_region_mappings = {}
        self.metrics = {}

    def process(self, experiment_name):
        raise NotImplemented





    def preprocess_voxels(self, scans):
        # Important: Order of preprocessing matters
        for voxel_preprocessing_fn, args in self.voxel_preprocessings:
            scans = voxel_preprocessing_fn(scans, **args)
        return scans

    def select_interesting_voxels(self, train_data, train_targets,
                                  number_of_selected_voxels=500):
        if self.voxel_selection == "none":
            return select_voxels_by_roi(train_targets)
        if self.voxel_selection == "roi":
            return select_voxels_by_roi(train_targets)

        # Data-driven selection on the training data
        else:
            # Mitchell calculates stability over several scans for the same word
            # if isinstance(self.brain_data_reader, MitchellReader):
            #     return self.brain_data_reader.get_stable_voxels(subject_id, test_stimuli)
            #
            # # As an alternative, we use the n voxels with the highest explained variance on the training data
            # else:

            varied_voxels = ignore_constant_voxels(train_targets)
            if self.task_is_text2brain:
                train_targets = self.reduce_voxels(train_targets, varied_voxels)
            else:
                train_data = self.reduce_voxels(train_data, varied_voxels)

            self.mapper.train(train_data, train_targets)

            train_predictions = self.mapper.map(inputs=train_data)["predictions"]

            predictive_voxels = select_voxels_by_variance(train_predictions, train_targets,
                                                          number_of_selected_voxels)
            # return intersection
            return sorted(list(set(varied_voxels) & set(predictive_voxels)))


    def reduce_voxels(self, list_of_scans, interesting_voxel_ids):
        reduced_list = np.asarray([list(scan[interesting_voxel_ids]) for scan in list_of_scans])

        return reduced_list

    def evaluate_fold(self, predictions, targets):
        logging.info("Evaluating...")
        results ={}
        for metric_name, metric_fn in self.metrics.items():
            results[metric_name] = metric_fn(predictions, targets)
        return results

    def save_evaluation(self, experiment_name, subject_id, collected_results):
        evaluation_file = self.save_dir + experiment_name + "/evaluation_" + str(
            subject_id) + "_standard_cv .txt"
        os.makedirs(os.path.dirname(evaluation_file), exist_ok=True)
        with open(evaluation_file, "w") as eval_file:
            eval_file.write(experiment_name + "\n")
            eval_file.write(str(subject_id) + "\n")
            for key, values in collected_results.items():
                eval_file.write(str(key) + "\t")
                for value in values:
                    eval_file.write(str(value) + "\t")
                eval_file.write("\n")


