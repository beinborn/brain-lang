"""Pipeline for predicting brain activations.
"""

from evaluation.metrics import *
from voxel_preprocessing.preprocess_voxels import detrend
from voxel_preprocessing.preprocess_voxels import minus_average_resting_states
from encoding_pipeline.Pipeline import Pipeline
import numpy as np
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)


class ContinuousPipeline(Pipeline):

    def __init__(self, brain_data_reader, stimuli_encoder, mapper, save_dir="processed_data/"):
        super(ContinuousPipeline, self).__init__(brain_data_reader, stimuli_encoder, mapper, save_dir)
        self.metrics = {"Average r2 score": mean_r2, "Sum r2 score": sum_r2, "Mean r2 for top 50": mean_r2_for_topn}
        self.voxel_preprocessings = [(detrend, {'t_r': 2.0})]
        # Delay is measured in TRs, a delay of 2 means that we align stimulus 1 with scan 3.
        self.delay = 2

    def process(self, experiment_name):

        # Reading data
        self.prepare_data(False, experiment_name)
        # Iterate over subjects
        for subject_id in self.subject_ids:
            print("Start processing for SUBJECT: " + str(subject_id))
            subject_data = self.data[subject_id]
            self.evaluate_crossvalidation(subject_id, subject_data, experiment_name)

    def evaluate_crossvalidation(self, subject_id, subject_blocks, experiment_name):
        collected_results = {}
        num_blocks = len(subject_blocks.keys())
        if num_blocks >= 2:
            testkeys = subject_blocks.keys()
            all_blocks = subject_blocks
        else:
            testkeys, all_blocks = self.split_data(subject_blocks)

        for testkey in testkeys:
            train_data, train_targets, test_data, test_targets = self.prepare_fold(testkey, all_blocks)
            logging.info("Select voxels on the training data")
            selected_voxels = self.select_interesting_voxels(train_data, train_targets, subject_id)

            # Reduce the scans and the predictions to the interesting voxels
            train_targets = self.reduce_voxels(train_targets, selected_voxels)
            test_targets = self.reduce_voxels(test_targets, selected_voxels)

            logging.info('Training completed. Training loss: ')
            self.mapper.train(train_data, train_targets)

            test_predictions = self.mapper.map(inputs=test_data, targets=test_targets)["predictions"]

            current_results = self.evaluate_fold(test_predictions, test_targets)

            print("Results for current fold: ")
            print(str(current_results))

            collected_results = self.update_collected_results(current_results, collected_results)
            print("Results for all folds")
            print(str(collected_results))

            # TODO calculate mean over collected results, save both
            logging.info("Writing evaluation")
            self.save_evaluation(experiment_name, subject_id, collected_results)
            logging.info("Experiment completed.")

    def prepare_fold(self, testblock, subject_blocks):
        train_scans = []
        train_embeddings = []
        for key, value in subject_blocks.items():
            scans = value[0]
            embeddings = value[1]
            if not key == testblock:
                train_scans += scans
                train_embeddings += embeddings
            else:
                test_scans = scans
                test_embeddings = embeddings

        train_scans = self.preprocess_voxels(train_scans)
        test_scans = self.preprocess_voxels(test_scans)

        # We could have a parameter here to switch scans and embeddings around for the decoding task
        train_data = train_embeddings
        test_data = test_embeddings
        train_targets = train_scans
        test_targets = test_scans

        return np.asarray(train_data), np.asarray(train_targets), np.asarray(test_data), np.asarray(
            test_targets)

        # If the data consists only of a single block, we need to find a way to split it into folds.
        # For the Alice data, we manually determined good break points which are hard-coded in the reader.

    def split_data(self, data):
        breakpoints = self.brain_data_reader.block_splits
        stimulus_number = 0
        splitted_data = {}
        block_number = 1
        scans = []
        embeddings = []

        # At this point, fixation scans have already been removed from the data
        # Scans and embeddings have already been aligned with the delay.
        # Breakpoints indicate sentence boundaries. We split the data into blocks without splitting a sentence.
        # Still, this is not optimal because the fMRI experiment was run as one continuous block.
        all_scans = data[1][0]
        all_embeddings = data[1][1]
        for i in range(0, len(all_scans)):

            if stimulus_number in breakpoints:
                splitted_data[block_number] = (scans, embeddings)
                print(len(scans), len(embeddings))
                scans = []
                embeddings = []
                block_number += 1

            scans.append(all_scans[i])
            embeddings.append(all_embeddings[i])
            stimulus_number += 1
        return splitted_data.keys(), splitted_data

    def prepare_data(self, normalize_by_rest, experiment_name):
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
                    sentences = block.sentences
                    logging.info("Get sentence embeddings")
                    sentence_embeddings = self.stimuli_encoder.get_sentence_embeddings(
                        experiment_name + "_" + str(block.block_id), sentences)
                    logging.info("Sentence embeddings obtained")
                    scans = [event.scan for event in block.scan_events]

                    stimulus_pointers = [event.stimulus_pointers for event in block.scan_events]
                    stimulus_embeddings = self.extract_stimulus_embeddings(sentence_embeddings, stimulus_pointers)
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

    def extract_stimulus_embeddings(self, sentence_embeddings, stimulus_pointers):
        stimulus_embeddings = []
        for stimulus_pointer in stimulus_pointers:
            if len(stimulus_pointer) > 0:
                token_embeddings = []
                for (sentence_id, token_id) in stimulus_pointer:
                    token_embeddings.append(sentence_embeddings[sentence_id][token_id])

                if len(token_embeddings) > 1:
                    # Instead of taking the mean here, other combinations like concatenation or sum could also work.
                    stimulus_embeddings.append(np.mean(token_embeddings, axis=0))
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

        # I am simply deleting scans with empty stimuli so far, there might be a better solution.
        embeddings = embeddings[i:]
        if normalize_by_resting:
            trial_scans = minus_average_resting_states(trial_scans, resting_scans)

        for i in range(0, len(trial_scans)):
            if (i + delay) < len(embeddings) and not len(embeddings[i + delay]) == 0:
                embedding = embeddings[i + delay]
                aligned_scans.append(trial_scans[i])
                aligned_embeddings.append(embedding)

        if not len(aligned_embeddings) == len(aligned_scans):
            raise ValueError(
                "Embedding length: " + str(len(aligned_embeddings)) + " Scan length: " + str(len(aligned_scans)))

        return aligned_scans, aligned_embeddings

    # TODO: can this go to the base class?
    def update_collected_results(self, current_results, collected_results):
        for key, value in current_results.items():

            if key in collected_results.keys():
                results = collected_results[key]
                results.append(value)
                collected_results[key] = results
            else:
                collected_results[key] = [value]
        return collected_results
