"""Pipeline for predicting brain activations.
"""

from evaluation.metrics import *

from voxel_preprocessing.select_voxels import *
from read_dataset.read_words_data import WordsReader
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
        all_voxels = len(scans[0])
        varied_voxels = ignore_constant_voxels(scans)
        #print("VARIED VOXELS: " + str(varied_voxels))
        #print(all_voxels, len(varied_voxels))
        selected_voxels = varied_voxels
        if voxel_selection =="none":
            logging.info("Not applying any selection. Just excluding the constant voxels. Number of kept voxels: " +str(len(varied_voxels)))
            return varied_voxels

        if voxel_selection == "roi":
            return select_voxels_by_roi(scans)

        if voxel_selection == "random":
            logging.info("Selecting voxels RANDOMLY")
            random.seed = 5
            random.shuffle(selected_voxels)
            selected_voxels = selected_voxels[:number_of_selected_voxels]
            logging.info("Voxel selection completed. Selected: " + str(len(selected_voxels)))
            return selected_voxels

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
            if (len(varied_voxels) < all_voxels):

                #selected_voxels = np.asarray(selected_voxels)[:, varied_voxels]
                selected_voxels= [voxel for voxel in selected_voxels if voxel in set(varied_voxels)]

            # selected_voxels = sorted(set(varied_voxels) & set(predictive_voxels), key=predictive_voxels.index)
            # return top n, list is sorted in ascending order so we take from the end

            topn_voxels = selected_voxels[-number_of_selected_voxels:]
            logging.info("Voxel selection completed. Selected: " + str(len(topn_voxels)))

            voxel_file = self.save_dir + "/" + self.pipeline_name + "/" + experiment_name + "selected_voxels.txt"
            with open(voxel_file, "a") as savefile:
                savefile.write(str(topn_voxels))
                savefile.write("\n")
            return topn_voxels
        else:
            raise ValueError("Choose a voxel selection mode. You can use 'none'.")


    def get_predictive_voxels(self, metric_fn, stimuli, scans, threshold, experiment_name):
        # Take 80 % as train
        number_of_train_samples =  int(len(stimuli) * 0.8)

        self.mapper.train(stimuli[0:number_of_train_samples], scans[0:number_of_train_samples])
        predictions = self.mapper.map(inputs=stimuli[number_of_train_samples:])["predictions"]
        ev_per_voxel = metric_fn(scans[number_of_train_samples:], predictions, multioutput="raw_values")
        print(ev_per_voxel)
        all_voxels = len(scans[0])
        varied_voxels = set(ignore_constant_voxels(scans))
        predictive_voxels =np.argwhere(np.asarray(ev_per_voxel)>threshold).tolist()
        print(all_voxels)
        print(len(predictive_voxels))
        if len(varied_voxels) < all_voxels:
            predictive_voxels = [v[0] for v in predictive_voxels if v[0] in varied_voxels]
        print(len(predictive_voxels))
        if (len(predictive_voxels)>1):
            indices = [x[0] for x in predictive_voxels]
        else:
            indices =predictive_voxels
        print(indices)
        voxel_file = self.save_dir + "/" + self.pipeline_name + "/" + experiment_name + "predictive_voxels.txt"
        with open(voxel_file, "w") as savefile:
            savefile.write("Explained variance scores: ")
            savefile.write(str([str(ev) for ev in ev_per_voxel]))
            savefile.write("\n")
            savefile.write("Voxels with explained variance above threshold " + str(threshold) +":  ")
            savefile.write(str(indices))
            savefile.write("\n")

        return indices



