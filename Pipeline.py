"""Pipeline for predicting brain activations.
"""

from evaluation.metrics import mean_explain_variance

from voxel_preprocessing.preprocess_voxels import detrend
from voxel_preprocessing.preprocess_voxels import minus_average_resting_states
from language_preprocessing.tokenize import SpacyTokenizer

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pickle


class Pipeline(object):
    def __init__(self, brain_data_reader, stimuli_encoder, mapper, save_dir="processed_data/"):
        self.brain_data_reader = brain_data_reader
        self.stimuli_encoder = stimuli_encoder
        self.mapper = mapper
        self.subject_ids = [1]
        self.delay = 0
        self.data_blocks = {}
        self.task_is_text2brain = True
        self.voxel_preprocessings = []
        # [(detrend, {'t_r': 2.0})]
        self.metrics = {'mean_EV': mean_explain_variance}
        self.save_dir = save_dir
        # TODO: how should we preprocess voxels: per block? Train and test separately? Does this make sense?

    def process(self, load_previous=False):
        tf.logging.info("Reading data")
        name = self.save_dir + "data/delay" + str(self.delay) + "/aligned_data.pickle"
        if load_previous:
            tf.logging.info("Loading from " + name)
            with open(name, 'rb') as handle:
                self.data_blocks = pickle.load(handle)
        else:
            all_blocks = self.brain_data_reader.read_all_events(subject_ids=self.subject_ids)
            tf.logging.info("Preparing data")
            self.prepare_data(all_blocks)
            with open(name, 'wb') as handle:
                pickle.dump(self.data_blocks, handle)

        # Start cross-validation
        # TODO find solution for alternative splits, eg. don't know yet how to split the alice data
        num_blocks = len(self.data_blocks.keys())
        if num_blocks >= 3:
            for testblock in self.data_blocks.keys():

                tf.logging.info("Starting fold, testing on: " + str(testblock))
                train_scans = []
                train_embeddings = []
                for key, value in self.data_blocks.items():
                    scans = value[0]
                    embeddings = value[1]
                    if not key == testblock:
                        train_scans += scans
                        train_embeddings += embeddings
                    else:
                        test_scans = scans
                        test_embeddings = embeddings


                # TODO how to deal with voxel selection?
                train_scans = self.preprocess_voxels(train_scans)
                test_scans = self.preprocess_voxels(test_scans)

                # Train the mapper
                tf.logging.info('Start training ...')
                # TODO I need to store the results somewhere to average over all folds
                if self.task_is_text2brain:
                    tf.logging.info("Training to predict brain activations")
                    train_input = np.asarray(train_embeddings)
                    train_targets = np.asarray(train_scans)
                    test_input = np.asarray(test_embeddings)
                    test_targets = np.asarray(test_scans)
                else:
                    train_input = np.asarray(train_scans)
                    train_targets = np.asarray(train_embeddings)
                    test_input = np.asarray(test_scans)
                    test_targets = np.asarray(test_embeddings)
                    tf.logging.info("Training to predict text embeddings")

                self.mapper.train(train_input, train_targets)
                tf.logging.info('Training completed.')
                print('Training loss: ')
                self.mapper.map(train_input, train_targets)
                tf.logging.info('Predicting voxel activations for test.')
                print(len(test_input))
                predictions = self.mapper.map(inputs=test_input, targets = test_targets)["predictions"]

                print(len(predictions))
                for metric_name, metric_fn in self.metrics.items():
                    metric_eval = metric_fn(predictions, test_targets)
                    print(metric_name, ":", metric_eval)
                tf.logging.info("End of fold")

    def prepare_data(self, reader_output, normalize_by_rest=False):
        for subject in reader_output.keys():
            blocks = reader_output[subject]
            for block in blocks:
                tf.logging.info("Preparing block " + str(block.block_id))
                # need to figure out how to apply tokenization and keep the stimuli intact
                # -> check how Samira did it, might also do it in the reader
                sentences = block.sentences
                tf.logging.info("Get sentence embeddings")
                sentence_embeddings = self.stimuli_encoder.get_sentence_embeddings(sentences, layer_id=1,
                                                                                   only_forward=False,
                                                                                   save_to=self.save_dir + "embeddings/block" + str(
                                                                                       block.block_id) + "/sentence_embeddings.pickle")
                tf.logging.info("Sentence embeddings obtained")
                scans = [event.scan for event in block.scan_events]

                stimulus_pointers = [event.stimulus_pointers for event in block.scan_events]
                stimulus_embeddings = self.extract_stimulus_embeddings(sentence_embeddings, stimulus_pointers, np.mean)
                if not len(scans) == len(stimulus_embeddings):
                    raise ValueError("")
                tf.logging.info("Align data with delay")
                self.data_blocks[block.block_id] = self.align_representations(scans, stimulus_embeddings, self.delay,
                                                                     normalize_by_rest)
                tf.logging.info("Data is aligned")

    def extract_stimulus_embeddings(self, sentence_embeddings, stimulus_pointers, integration_fn=np.mean):
        stimulus_embeddings = []
        for stimulus_pointer in stimulus_pointers:
            if len(stimulus_pointer) > 0:
                token_embeddings = []
                for (sentence_id, token_id) in stimulus_pointer:
                    token_embeddings.append(sentence_embeddings[sentence_id][token_id])
                # TODO I don't yet see whether this would work if integration_fn is something else
                stimulus_embeddings.append(integration_fn(token_embeddings, axis=0))
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
        embeddings =embeddings[i:]
        if normalize_by_resting:
            trial_scans = minus_average_resting_states(trial_scans, resting_scans)

        for i in range(0, len(trial_scans)):
            if (i + delay) < len(embeddings) and not len(embeddings[i+delay]) ==0:
                embedding = embeddings[i + delay]
                # print("Aligning scan " + str(i) + " and embedding " + str(i + delay))
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
