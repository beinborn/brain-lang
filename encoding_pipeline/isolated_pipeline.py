from encoding_pipeline.pipeline_utils import *
from encoding_pipeline.pipeline import Pipeline
from evaluation.metrics import *
from evaluation.evaluation_util import *
import numpy as np
import datetime
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)


# This class runs experiments on datasets with isolated stimuli.
# Isolated = One (averaged) scan per stimulus, stimuli are not related, order of stimuli does not matter.
# Note: if a pickle with data is found in the save_dir, it is automatically used.
# Make sure to delete it or change the name of the save dir, if you want to rerun everything.
class SingleInstancePipeline(Pipeline):
    def __init__(self, brain_data_reader, stimuli_encoder, mapper, pipeline_name, save_dir="processed_data/"):
        super(SingleInstancePipeline, self).__init__(brain_data_reader, stimuli_encoder, mapper, pipeline_name,
                                                     save_dir)
        self.data = {}
        self.metrics = {"R2": r2_score_complex, "Explained Variance": explained_variance_complex,
                        "Pearson_squared": pearson_jain_complex}
        self.save_dir = save_dir

        # Example for preprocessing: [(detrend, {'t_r': 2.0})]
        # Options for voxel selection are: "on_train_r", "on_train_ev"random", "by_roi", "stable" and "none"
        self.voxel_preprocessings = []
        self.voxel_selection = "on_train_ev"
        self.roi = []
        self.voxel_to_region_mappings = {}

    # Run standard cross validation over all subjects
    def run_standard_crossvalidation(self, experiment_name):
        # Reading data
        self.prepare_data()

        # Set subjects
        if self.subject_ids == None:
            self.subject_ids = self.data.keys()
        logging.info("Subjects: " + str(self.subject_ids))

        # Iterate over subjects
        for subject_id in self.subject_ids:
            logging.info("Start processing for SUBJECT: " + str(subject_id))
            subject_data = self.data[subject_id]
            self.evaluate_crossvalidation(subject_id, subject_data, experiment_name)

    # Run the pair-wise evaluation procedure described in Mitchell et al. (2008)
    # Note: this can take long.
    def run_pairwise_procedure(self, experiment_name):
        # Reading data
        self.prepare_data()

        # Iterate over subjects
        if self.subject_ids == None:
            self.subject_ids = self.data.keys()
        print("Subjects: " + str(self.subject_ids))
        for subject_id in self.subject_ids:
            print("Start processing for SUBJECT: " + str(subject_id))
            subject_data = self.data[subject_id]
            self.evaluate_leave_two_out_procedure(subject_id, subject_data, experiment_name)

    # This method runs representational similarity analysis as described by Kriegeskorte et al. (2006)
    def runRSA(self, experiment_name):

        # Reading data
        self.prepare_data()
        # Iterate over subjects

        if self.subject_ids == None:
            self.subject_ids = list(self.data.keys())
        print("Subjects: " + str(self.subject_ids))

        print(self.data.keys())
        scores = {}
        for subject_id in self.subject_ids:
            print("Start processing for SUBJECT: " + str(subject_id))

            # Iterate through blocks and collect all scans and embeddings
            all_scans, all_embeddings, _ = self.data[subject_id]

            # Preprocess voxels
            all_scans = preprocess_voxels(self.voxel_preprocessings, np.asarray(all_scans))
            # TODO might want to add voxel selection here
            # Collect data from subjects

            x, C = get_dists([all_scans, all_embeddings])

            spearman, pearson, kullback = compute_distance_over_dists(x, C)

            score = (spearman[0][1], pearson[0][1], kullback[0][1])
            scores[subject_id] = score
            print("Spearman, Pearson, Kullback: ")
            print(score)
        all_scores = scores.values()
        mean_spearman = np.mean(np.asarray([score[0] for score in all_scores]))
        mean_pearson = np.mean(np.asarray([score[1] for score in all_scores]))
        mean_kullback = np.mean(np.asarray([score[2] for score in all_scores]))
        print("Means: Spearman, Pearson, Kullback")
        print(mean_spearman, mean_pearson, mean_kullback)
        rdm_file = self.save_dir + self.pipeline_name + "/" + experiment_name + "_RDM.txt"
        with open(rdm_file, "w") as eval_file:
            eval_file.write("Subject\tSpearman\t,Pearson\t,Kullback\n")
            for subject in scores.keys():
                result = scores[subject]
                eval_file.write(
                    str(subject) + "\t" + str(result[0]) + "\t" + str(result[1]) + "\t" + str(result[2]) + "\n")
            eval_file.write(
                "\nAverages:" + "\t" + str(mean_spearman) + "\t" + str(mean_pearson) + "\t" + str(mean_kullback) + "\n")

    def evaluate_crossvalidation(self, subject_id, subject_data, experiment_name):
        # Determine folds so that we have 10 stimuli per fold
        # We do not use 10 folds because folds should be bigger to evaluate explained variance
        number_of_folds = int(len(subject_data[0]) / 10)
        collected_results = {}

        for fold in range(0, number_of_folds):
            start = fold * 10
            end = fold * 10 + 10
            testids = list(range(start, end))
            logging.info("Starting fold: " + str(fold))
            test_stimuli = [subject_data[2][i] for i in testids]
            print(testids, test_stimuli)

            # Split data into folds
            train_data, train_targets, test_data, test_targets = self.prepare_fold(subject_data, testids)

            # Select voxels on train
            logging.info("Select voxels on the training data")
            selected_voxels = self.select_interesting_voxels(self.voxel_selection, train_data, train_targets,
                                                             experiment_name, 500)
            logging.info("Number of selected voxels: " + str(len(selected_voxels)))

            # Reduce the scans and the predictions to the interesting voxels
            logging.info('Voxel selection completed.')
            train_targets = reduce_voxels(train_targets, selected_voxels)
            test_targets = reduce_voxels(test_targets, selected_voxels)

            # Train the mapper
            self.mapper.train(train_data, train_targets)
            logging.info('Training completed. Training loss: ')

            # Get predictions
            test_predictions = self.mapper.map(inputs=test_data, targets=test_targets)["predictions"]

            # Get evaluation scores
            current_results = evaluate_fold(self.metrics, test_predictions, test_targets)
            print("Results for current fold: ")
            print(str(current_results["R2"][1]))
            collected_results = update_results(current_results, collected_results)

        # Write out results
        print("Results for all folds")
        evaluation_path = self.save_dir + self.pipeline_name + "/evaluation_" + str(
            subject_id) + experiment_name + "_standard_cv.txt"
        logging.info("Writing evaluation to " + evaluation_path)
        save_evaluation(evaluation_path, experiment_name, subject_id, collected_results)

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
                if (self.voxel_selection == "stable"):
                    selected_voxels = self.brain_data_reader.get_stable_voxels_for_fold(subject_id, test_stimuli)
                    print(selected_voxels)
                else:
                    selected_voxels = self.select_interesting_voxels(self.voxel_selection, train_data, train_targets,
                                                                     experiment_name)
                print("Number of selected voxels: " + str(len(selected_voxels)))
                # Reduce the scans and the predictions to the interesting voxels

                train_targets = reduce_voxels(train_targets, selected_voxels)
                test_targets = reduce_voxels(test_targets, selected_voxels)
                pairs += 1
                logging.info('Training completed. Training loss: ')
                self.mapper.train(train_data, train_targets)

                test_predictions = self.mapper.map(inputs=test_data, targets=test_targets)["predictions"]
                if not len(test_predictions) == 2:
                    raise RuntimeError("Something went wrong with the prediction")

                matches = pairwise_matches(test_predictions[0], test_targets[0],
                                           test_predictions[1], test_targets[1])
                collected_matches = add_to_collected_results(matches, collected_matches)
                print(str(collected_matches))

        evaluation_file = self.save_dir + self.pipeline_name + "/evaluation_" + str(
            subject_id) + experiment_name + "_2x2.txt"
        logging.info("Writing evaluation to " + evaluation_file)
        save_pairwise_evaluation(evaluation_file, self.pipeline_name, subject_id, collected_matches, pairs)

        logging.info("Experiment completed.")


    # This method determines the voxels with the best prediction results (above threshold) when training on 80% and testing on 20%
    def get_all_predictive_voxels(self, experiment_name, threshold = 0.4):

        # Reading data
        self.prepare_data()

        # Make sure that voxel selection is none
        self.voxel_selection = "none"
        # Iterate over subjects
        print("Subjects: " + str(self.data.keys()))
        if self.subject_ids == None:
            self.subject_ids = list(self.data.keys())
        for subject_id in self.subject_ids:
            print("Start processing for SUBJECT: " + str(subject_id))

            all_scans = []
            all_embeddings = []

            # Collect all scans and embeddings
            scans, embeddings, _ = self.data[subject_id]
            all_scans.extend(scans)
            all_embeddings.extend(embeddings)

            # Take 80 % as train
            number_of_train_samples = int(len(embeddings) * 0.8)
            train_data = embeddings[0:number_of_train_samples]
            train_targets = scans[0:number_of_train_samples]
            test_data = embeddings[number_of_train_samples:]
            test_targets = scans[number_of_train_samples:]
            predictive_voxels = self.get_predictive_voxels(explained_variance_score, train_data, train_targets, test_data, test_targets, threshold,
                                                           experiment_name + "_" + str(subject_id))
            print(predictive_voxels)
            return predictive_voxels



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

        # We are preprocessing the voxels separately for every fold
        train_scans = preprocess_voxels(self.voxel_preprocessings, train_scans)
        test_scans = preprocess_voxels(self.voxel_preprocessings, test_scans)

        # We could have a parameter here to switch scans and embeddings around for the decoding task
        train_data = train_embeddings
        test_data = test_embeddings
        train_targets = train_scans
        test_targets = test_scans

        return np.asarray(train_data), np.asarray(train_targets), np.asarray(test_data), np.asarray(test_targets)

    # This method reads the data, gets the embeddings and saves everything.
    def prepare_data(self):
        datafile = self.save_dir + self.pipeline_name + "/aligned_data.pickle"
        if os.path.isfile(datafile):
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

                # Distinguish between getting embeddings for words and for stories.
                # Note: For other datasets, there might be the need to distinguish between story and sentence embeddings.
                if max([len(x[0]) for x in stimuli]) == 1:
                    stimuli = [word[0] for word in stimuli]
                    print(stimuli)
                    embeddings = self.stimuli_encoder.get_word_embeddings(
                        self.pipeline_name + "/embeddings/" + str(subject) + "_", stimuli)
                else:
                    embeddings = self.stimuli_encoder.get_story_embeddings(
                        self.pipeline_name + "/embeddings/" + str(subject) + "_", stimuli)

                logging.info("Embeddings obtained")

                self.voxel_to_region_mappings[subject] = all_blocks[subject][0].voxel_to_region_mapping
                # self.voxel_to_xyz_mapping[subject] = block.voxel_to_xyz_mapping

                if not len(scans) == len(embeddings):
                    raise ValueError("Lengths disagree, scans: " + str(len(scans)) + " embeddings: " + str(
                        len(embeddings)))
                self.data[subject] = (scans, embeddings, stimuli)
                print(np.asarray(embeddings).shape)

                # Save data
                os.makedirs(os.path.dirname(datafile), exist_ok=True)
                with open(datafile, 'wb') as handle:
                    pickle.dump(self.data, handle)
