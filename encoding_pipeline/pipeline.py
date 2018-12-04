"""Pipeline for predicting brain activations.
"""

from evaluation.metrics import *

from voxel_preprocessing.select_voxels import *
from read_dataset.read_words_data import MitchellReader
import numpy as np
from datetime import datetime
import pickle
import os
import logging
import random

logging.basicConfig(level=logging.INFO)


class Pipeline(object):
    def __init__(self, brain_data_reader, stimuli_encoder, mapper, pipeline_name, save_dir="processed_data/"):
        self.brain_data_reader = brain_data_reader
        self.stimuli_encoder = stimuli_encoder
        self.mapper = mapper
        self.subject_ids = None
        self.pipeline_name = pipeline_name


    def process(self):
        raise NotImplemented


    def select_interesting_voxels(self, voxel_selection, stimuli, scans, experiment_name,
                                  number_of_selected_voxels=500):
        # Default: only ignore constant voxels
        all_voxels = [range(0, len(scans[0]))]
        varied_voxels = ignore_constant_voxels(scans)

        selected_voxels = varied_voxels
        if voxel_selection =="none":
            logging.info("Not applying any selection. Just excluding the constant voxels. Number of kept voxels: " +str(len(varied_voxels)))
            return varied_voxels
        else:
            if voxel_selection == "roi":
                selected_voxels = select_voxels_by_roi(scans)

            if voxel_selection == "random":
                logging.info("Selecting voxels RANDOMLY")
                random.seed = 5
                selected_voxels = random.shuffle(selected_voxels)[:number_of_selected_voxels]
            # Data-driven selection on the training data
            if voxel_selection.startswith("on_train"):
                self.mapper.train(stimuli, scans)
                logging.info("Voxel evaluation model is trained.")

                train_predictions = self.mapper.map(inputs=stimuli)["predictions"]
                if self.voxel_selection.endswith("r"):
                    logging.info("Evaluate voxels BY R")
                    selected_voxels = select_voxels_by_r(train_predictions, scans)
                else:
                    logging.info("Evaluate voxels BY EV")
                    selected_voxels = select_voxels_by_variance(train_predictions, scans)
                logging.info("Voxels are evaluated")

                # take intersection of predictive and non-constant voxels
                if (len(varied_voxels) < len(all_voxels)):
                    print(varied_voxels)
                    print(len(selected_voxels), len(varied_voxels))
                    print(selected_voxels[0:10])
                    selected_voxels = np.asarray(selected_voxels)[:, varied_voxels]
                    print(selected_voxels[0:10])
                    # selected_voxels= [voxel for voxel in predictive_voxels if voxel in varied_voxels]
                # selected_voxels = sorted(set(varied_voxels) & set(predictive_voxels), key=predictive_voxels.index)
                # return top n, list is sorted in ascending order so we take from the end
                topn_voxels = selected_voxels[-number_of_selected_voxels:]
                logging.info("Voxel selection completed. Selected: " + str(len(selected_voxels)))

            voxel_file = self.save_dir + "/" + self.pipeline_name + "/" + experiment_name + "selected_voxels.txt"
            with open(voxel_file, "a") as savefile:
                savefile.write(str(selected_voxels))

            return topn_voxels





