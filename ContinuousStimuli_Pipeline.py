"""Pipeline for predicting brain activations.
"""

from evaluation.metrics import *

from voxel_preprocessing.preprocess_voxels import detrend
from voxel_preprocessing.preprocess_voxels import minus_average_resting_states
from voxel_preprocessing.select_voxels import select_voxels_by_roi
from language_preprocessing.tokenize import SpacyTokenizer

import tensorflow as tf
import numpy as np
from tqdm import tqdm
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
        self.delay = 0
        self.data = {}
        self.task_is_text2brain = True
        self.voxel_preprocessings = []
        # [(detrend, {'t_r': 2.0})]
        self.metrics = {'Average explained variance': mean_explained_variance, "2x2 accuracy": binary_accuracy}
        self.save_dir = save_dir
        self.load_previous = False
        self.voxel_to_region_mappings = {}
        # TODO: how should we preprocess voxels: per block? Train and test separately? Does this make sense?

    def process(self, experiment_name):

        # Reading data
        self.prepare_data(False)

        # Iterate over subjects
        for subject_id in self.data.keys():
            print("Start processing for SUBJECT: " + str(subject_id))
            subject_blocks = self.data[subject_id]
            num_blocks = len(subject_blocks.keys())
            if num_blocks >= 2:
                logging.info("Collect predictions per block")
                predictions, test_targets = self.get_predictions_per_block(subject_blocks)
            else:
                # We cannot split the alice data into blocks, so we do cross-validation over scans.
                # Does not seem very smart, but I couldn't come up with a better solution.
                logging.info("Collect predictions per scan")
                predictions, test_targets = self.get_predictions_per_scan(subject_blocks[1])

            # TODO: add voxel selection here: ROI-based, search-light based
            # The mapping model treats all voxels independently (which is not very reasonable).
            # So we can do voxel selection after learning the model

            logging.info("End of cross-validation. Evaluate all predictions")
            interesting_voxel_ids = self.get_interesting_voxel_ids(predictions, test_targets, subject_id)
            print("Length of predictions: " + str(len(predictions[0])))
            #for selection in interesting_voxel_ids:
             #   predictions = predictions[selection]
             #   test_targets = test_targets[selection]
            print("Length of selected predictions: " + str(len(predictions[0])))

            # TODO add method for searchlight analysis
            # TODO write evaluation to save_dir/experiment_name
            self.evaluate(np.asarray(predictions),
                          np.asarray(test_targets))

    def get_predictions_per_block(self, blocks):
        # Start cross-validation
        all_predictions = []
        all_targets = []

        for testblock in blocks:
            logging.info("Starting fold, testing on: " + str(testblock))
            train_scans = []
            train_embeddings = []
            for key, value in self.data.items():
                scans = value[0]
                embeddings = value[1]
                if not key == testblock:
                    train_scans += scans
                    train_embeddings += embeddings
                else:
                    test_scans = scans
                    test_embeddings = embeddings

            # TODO is that the right place for voxel preprocessing?
            train_scans = self.preprocess_voxels(train_scans)
            test_scans = self.preprocess_voxels(test_scans)

            # Train the mapper
            logging.info('Start training ...')
            if self.task_is_text2brain:
                logging.info("Training to predict brain activations")
                self.mapper.train(train_embeddings, train_scans)
                logging.info('Training completed. Training loss: ')
                predictions = self.mapper.map(inputs=test_embeddings, targets=test_scans)["predictions"]
                all_targets.extend(test_scans)
            else:
                logging.info("Training to predict text embeddings")
                self.mapper.train(train_scans, train_embeddings)
                logging.info('Training completed. Training loss: ')
                predictions = self.mapper.map(inputs=test_scans, targets=test_embeddings)["predictions"]
                all_targets.extend(test_embeddings)

            all_predictions.extend(predictions)

        return np.asarray(all_predictions), np.asarray(all_targets)


    def get_predictions_per_scan(self, block):
        # Start cross-validation
        all_predictions = []
        all_targets = []



        for id in range(0, len(block[0])):
            test_scans = [block[0][id]]
            test_embeddings = [block[1][id]]

            train_scans = []
            train_embeddings = []
            for x in range(0, len(block)):
                if not x == id:
                    train_scans.append(block[0][x])
                    train_embeddings.append(block[1][x])

            # TODO is that the right place for voxel preprocessing?
            train_scans = self.preprocess_voxels(train_scans)
            test_scans = self.preprocess_voxels(test_scans)

            # Train the mapper
            logging.info('Start training ...')

            if self.task_is_text2brain:
                logging.info("Training to predict brain activations")
                self.mapper.train(train_embeddings, train_scans)
                logging.info('Training completed. Training loss: ')
                predictions = self.mapper.map(inputs=test_embeddings)["predictions"]

                all_targets.extend(test_scans)
            else:
                logging.info("Training to predict text embeddings")
                self.mapper.train(train_scans, train_embeddings)
                logging.info('Training completed. Training loss: ')
                predictions = self.mapper.map(inputs=test_scans)["predictions"]
                all_targets.extend(test_embeddings)

            all_predictions.extend(predictions)

        return np.asarray(all_predictions), np.asarray(all_targets)


    # TODO: this does not yet work for more than 1 subject!!!
    def prepare_data(self, normalize_by_rest):
        datafile = self.save_dir + "data/delay" + str(self.delay) + "/aligned_data.pickle"

        if self.load_previous:
            logging.info("Loading from " + datafile)
            with open(datafile, 'rb') as handle:
                self.data = pickle.load(handle)
        else:
            all_blocks = self.brain_data_reader.read_all_events(subject_ids=self.subject_ids)

            for subject in all_blocks.keys():
                blocks = all_blocks[subject]
                self.data[subject] = {}
                logging.info("Preparing data for subject " + str(subject))
                for block in blocks:
                    logging.info("Preparing block " + str(block.block_id))

                    self.voxel_to_region_mappings[subject] = block.voxel_to_region_mapping
                    #self.voxel_to_xyz_mapping[subject] = block.voxel_to_xyz_mapping
                    sentences = block.sentences
                    logging.info("Get sentence embeddings")
                    sentence_embeddings = self.stimuli_encoder.get_sentence_embeddings(block.block_id, sentences)
                    logging.info("Sentence embeddings obtained")
                    scans = [event.scan for event in block.scan_events]

                    stimulus_pointers = [event.stimulus_pointers for event in block.scan_events]
                    stimulus_embeddings = self.extract_stimulus_embeddings(sentence_embeddings, stimulus_pointers,
                                                                           np.mean)
                    if not len(scans) == len(stimulus_embeddings):
                        raise ValueError("Lengths disagree, scans: " + str(len(scans)) + " embeddings: " + str(
                            len(stimulus_embeddings)))
                    logging.info("Align data with delay")
                    self.data[subject][block.block_id] = self.align_representations(scans, stimulus_embeddings,
                                                                                    self.delay,
                                                                                    normalize_by_rest)
                    logging.info("Data is aligned")

                    os.makedirs(os.path.dirname(datafile), exist_ok=True)
                    with open(datafile, 'wb') as handle:
                        pickle.dump(self.data, handle)


    def extract_stimulus_embeddings(self, sentence_embeddings, stimulus_pointers, integration_fn=np.mean):
        stimulus_embeddings = []
        for stimulus_pointer in stimulus_pointers:
            if len(stimulus_pointer) > 0:
                token_embeddings = []
                for (sentence_id, token_id) in stimulus_pointer:
                    token_embeddings.append(sentence_embeddings[sentence_id][token_id])
                # TODO I don't yet see whether this would work if integration_fn is something else
                if len(token_embeddings) > 1:
                    stimulus_embeddings.append(integration_fn(token_embeddings, axis=0))
                else:
                    stimulus_embeddings.append(token_embeddings[0])
            else:
                # I am returning the empty embedding if we do not have a stimulus. There might be smarter ways.
                stimulus_embeddings.append([])

        return stimulus_embeddings


    def align_representations(self, scans, embeddings, delay, normalize_by_resting):
        aligned_scans = []
        aligned_embeddings = []

        resting_scans = []
        trial_scans = []
        i = 0

        while len(embeddings[i]) == 0:
            resting_scans.append(scans[i])
            i += 1
        for scan in scans[i:]:
            trial_scans.append(scan)

        # TODO I am simply deleting scans with empty stimuli, there might be a better solution
        embeddings = embeddings[i:]
        if normalize_by_resting:
            trial_scans = minus_average_resting_states(trial_scans, resting_scans)

        for i in range(0, len(trial_scans)):
            if (i + delay) < len(embeddings) and not len(embeddings[i + delay]) == 0:
                embedding = embeddings[i + delay]
                # logging.info("Aligning scan " + str(i) + " and embedding " + str(i + delay))
                aligned_scans.append(trial_scans[i])
                aligned_embeddings.append(embedding)

        if not len(aligned_embeddings) == len(aligned_scans):
            raise ValueError(
                "Embedding length: " + str(len(aligned_embeddings)) + " Scan length: " + str(len(aligned_scans)))

        return aligned_scans, aligned_embeddings


    def preprocess_voxels(self, scans):
        # Important: Order of preprocessing matters
        for voxel_preprocessing_fn, args in self.voxel_preprocessings:
            scans = voxel_preprocessing_fn(scans, **args)
        return scans


    def get_interesting_voxel_ids(self, train_predictions, train_targets, subject_id):
        # TODO find solution to select selection method using a parameter
        interesting_voxel_ids = select_voxels_by_roi(train_targets, self.voxel_to_region_mapping[subject_id],
                                                     ["Temporal_Sup_L", "Temporal_Sup_R", "Temporal_Pole_Sup_L",
                                                      "Temporal_Pole_Sup_R", "Temporal_Mid_L", "Temporal_Mid_R",
                                                      "Temporal_Pole_Mid_L", "Temporal_Pole_Mid_R",
                                                      "Temporal_Inf_L", "Temporal_Inf_R"])
        # TODO: this needs to be selected from within the cross-validation
        # interesting_voxel_ids = get_topn_voxels(train_predictions, train_targets)
        print("Number of interesting voxels: " + str(len(interesting_voxel_ids)))
        return [interesting_voxel_ids]


    def evaluate(self, predictions, targets):
        # TODO this function should return something!
        logging.info("Evaluating...")
        for metric_name, metric_fn in self.metrics.items():
            metric_eval = metric_fn(predictions, targets)
            print(metric_name, ":", metric_eval)
