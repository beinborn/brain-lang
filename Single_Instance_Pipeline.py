"""Pipeline for predicting brain activations.
"""

from evaluation.metrics import *

from voxel_preprocessing.select_voxels import *
from read_dataset.readMitchellData import MitchellReader
import numpy as np

import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)


class SingleInstancePipeline(object):
    def __init__(self, brain_data_reader, stimuli_encoder, mapper, save_dir="processed_data/"):
        self.brain_data_reader = brain_data_reader
        self.stimuli_encoder = stimuli_encoder
        self.mapper = mapper
        self.subject_ids = [1]
        self.data = {}
        self.task_is_text2brain = True
        self.voxel_preprocessings = []
        # Example: [(detrend, {'t_r': 2.0})]
        self.voxel_selections = []
        self.roi = []
        self.save_dir = save_dir
        self.load_previous = False
        self.voxel_to_region_mappings = {}
        self.metrics = {'Average explained variance': mean_explained_variance, "Sum explained variance": sum_explained_variance,"Mean explained variance for top 50": mean_ev_for_topn, "Average r2 score": mean_r2, "Sum r2 score": sum_r2, "Mean r2 for top 50": mean_r2_for_topn}

    def process(self, experiment_name):
        # Reading data
        self.prepare_data(experiment_name)

        # Iterate over subjects
        for subject_id in self.data.keys():
            print("Start processing for SUBJECT: " + str(subject_id))
            subject_data = self.data[subject_id]
            # Careful! They call it cross-validation, but every scan is in the testset n times
            # n = number of scans
            self.evaluate_crossvalidation(subject_id, subject_data, experiment_name)
            self.evaluate_leave_two_out_procedure(subject_id, subject_data, experiment_name)



    def evaluate_crossvalidation(self, subject_id, subject_data, experiment_name):
        # Split into folds so that we have 10 scans per fold
        # We do not make it 10 folds because folds should be bigger to evaluate explained variance
        number_of_folds = int(len(subject_data[0]) / 10)
        collected_results = {}

        for fold in range(0, number_of_folds):
            start = fold * 10
            end = fold * 10 + 10
            testids = list(range(start, end))
            print("Starting fold: " + str(fold))
            print(testids)
            # Not sure if this works
            test_stimuli = [subject_data[2][i] for i in testids]
            print(testids, test_stimuli)
            train_data, train_targets, test_data, test_targets = self.prepare_fold(subject_data, testids)

            logging.info("Select voxels on the training data")
            selected_voxels = self.select_interesting_voxels(train_data, train_targets, subject_id,
                                                             test_stimuli)

            # Reduce the scans and the predictions to the interesting voxels
            if self.task_is_text2brain:
                train_targets = self.reduce_voxels(train_targets, selected_voxels)
                test_targets = self.reduce_voxels(test_targets, selected_voxels)
            else:
                train_data = self.reduce_voxels(train_data, selected_voxels)
                test_data = self.reduce_voxels(test_data, selected_voxels)

            logging.info('Training completed. Training loss: ')
            self.mapper.train(train_data, train_targets)

            test_predictions = self.mapper.map(inputs=test_data, targets=test_targets)["predictions"]

            current_results = self.evaluate_fold(test_predictions, test_targets)
            print("Results for current fold: ")
            print(str(current_results))
            for key, value in current_results.items():

                if key in collected_results.keys():
                    results = collected_results[key]
                    results.append(value)
                    collected_results[key] = results
                else:
                    collected_results[key] = [value]
        print("Results for all folds")
        print(str(collected_results))


        evaluation_file = self.save_dir + experiment_name +  "/evaluation_" + str(
            subject_id) + "_standard_cv .txt"
        logging.info("Writing evaluation to " + evaluation_file)
        os.makedirs(os.path.dirname(evaluation_file), exist_ok=True)
        with open(evaluation_file, "w") as eval_file:
            eval_file.write(experiment_name + "\n")
            eval_file.write(str(subject_id) + "\n")
            for key, values in collected_results.items():
                eval_file.write(str(key) + "\t")
                for value in values:
                    eval_file.write(str(value) + "\t")
                eval_file.write("\n")

        logging.info("Experiment completed.")



    def evaluate_leave_two_out_procedure(self, subject_id, subject_data, experiment_name):
        collected_matches = {}
        pairs = 0
        logging.info("Start leave-two out procedure")

        for i in range(0, len(subject_data[0])):
            for k in range(i + 1, len(subject_data[0])):
                logging.info("Evaluate on pair " + str(pairs))
                test_stimuli = [subject_data[2][i][0], subject_data[2][k][0]]
                print(test_stimuli)
                train_data, train_targets, test_data, test_targets = self.prepare_fold(subject_data, [i, k])
                if not len(test_data) == 2 or not len(test_targets) == 2:
                    raise RuntimeError(
                        "Something went wrong with preparing the test data: " + str(len(test_data)) + " " + str(
                            len(test_targets)))

                logging.info("Select voxels on the training data")
                selected_voxels = self.select_interesting_voxels(train_data, train_targets, subject_id,
                                                                 test_stimuli)

                # Reduce the scans and the predictions to the interesting voxels
                if self.task_is_text2brain:
                    train_targets = self.reduce_voxels(train_targets, selected_voxels)
                    test_targets = self.reduce_voxels(test_targets, selected_voxels)
                else:
                    train_data = self.reduce_voxels(train_data, selected_voxels)
                    test_data = self.reduce_voxels(test_data, selected_voxels)

                logging.info('Training completed. Training loss: ')
                self.mapper.train(train_data, train_targets)

                test_predictions = self.mapper.map(inputs=test_data, targets=test_targets)["predictions"]
                if not len(test_predictions) == 2:
                    raise RuntimeError("Something went wrong with the prediction")

                pairs += 1

                matches = pairwise_matches(test_predictions[0], test_targets[0],
                                           test_predictions[1], test_targets[1])

                for key, value in matches.items():

                    if key in collected_matches.keys():
                        previous_matches = collected_matches[key]
                        collected_matches[key] = value + previous_matches
                    else:
                        collected_matches[key] = value
                print(str(collected_matches))
        evaluation_file = self.save_dir +  experiment_name +"/evaluation_" + str(subject_id) + "_2x2.txt"
        logging.info("Writing evaluation to " + evaluation_file)
        os.makedirs(os.path.dirname(evaluation_file), exist_ok=True)
        with open(evaluation_file, "w") as eval_file:
            eval_file.write(experiment_name + "\n")
            eval_file.write(str(subject_id) + "\n")
            for key, value in collected_matches.items():
                eval_file.write(str(key) + "\t" + str(value) + "\n")

        logging.info("Experiment completed.")

    def prepare_fold(self, data, testids):
        train_scans = []
        train_embeddings = []
        test_scans = []
        test_embeddings = []
        for i in range(0, len(data[0])):
            scan = data[0][i]
            embedding = data[1][i]
            if i in testids:
                test_scans.append(scan)
                test_embeddings.append(embedding)
            else:
                train_scans.append(scan)
                train_embeddings.append(embedding)

        # TODO is that the right place for voxel preprocessing?
        train_scans = self.preprocess_voxels(train_scans)
        test_scans = self.preprocess_voxels(test_scans)

        if self.task_is_text2brain:
            train_data = train_embeddings
            test_data = test_embeddings
            train_targets = train_scans
            test_targets = test_scans
        else:
            train_data = train_scans
            test_data = test_scans
            train_targets = train_embeddings
            test_targets = test_embeddings
        return np.asarray(train_data), np.asarray(train_targets), np.asarray(test_data), np.asarray(test_targets)

    def prepare_data(self, name):
        datafile = self.save_dir + name + "/aligned_data.pickle"
        if self.load_previous:
            logging.info("Loading from " + datafile)
            with open(datafile, 'rb') as handle:
                self.data = pickle.load(handle)
        else:

            all_blocks = self.brain_data_reader.read_all_events(subject_ids=self.subject_ids)
            for subject in all_blocks.keys():
                logging.info("Preparing data for subject " + str(subject))
                stimuli = []
                scans = []
                for block in all_blocks[subject]:
                    stimuli.append(block.sentences)
                    scans.append(block.scan_events[0].scan)

                if max([len(x[0]) for x in stimuli]) == 1:
                    stimuli = [word[0] for word in stimuli]
                    print(stimuli)
                    embeddings = self.stimuli_encoder.get_word_embeddings(name, stimuli)
                else:
                    embeddings = self.stimuli_encoder.get_story_embeddings(name, stimuli)

                logging.info("Embeddings obtained")

                self.voxel_to_region_mappings[subject] = all_blocks[subject][0].voxel_to_region_mapping
                # self.voxel_to_xyz_mapping[subject] = block.voxel_to_xyz_mapping

                if not len(scans) == len(embeddings):
                    raise ValueError("Lengths disagree, scans: " + str(len(scans)) + " embeddings: " + str(
                        len(embeddings)))
                self.data[subject] = (scans, embeddings, stimuli)
                print(np.asarray(embeddings).shape)
                os.makedirs(os.path.dirname(datafile), exist_ok=True)
                with open(datafile, 'wb') as handle:
                    pickle.dump(self.data, handle)

    def preprocess_voxels(self, scans):
        # Important: Order of preprocessing matters
        for voxel_preprocessing_fn, args in self.voxel_preprocessings:
            scans = voxel_preprocessing_fn(scans, **args)
        return scans

    def select_interesting_voxels(self, train_data, train_targets, subject_id, test_stimuli,
                                  number_of_selected_voxels=500):
        # Hypothesis-driven selection
        if len(self.roi) > 0:
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
#     # TODO this function should return something!
        logging.info("Evaluating...")
        results ={}
        for metric_name, metric_fn in self.metrics.items():
            results[metric_name] = metric_fn(predictions, targets)
        return results
