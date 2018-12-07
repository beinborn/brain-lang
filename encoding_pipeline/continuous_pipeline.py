"""Pipeline for predicting brain activations.
"""

import datetime
from encoding_pipeline.pipeline_utils import *
from encoding_pipeline.pipeline import Pipeline
from evaluation.metrics import *
from evaluation.evaluation_util import *
import numpy as np
import pickle
import os
import logging
from result_analysis import plot_rdm
logging.basicConfig(level=logging.INFO)


class ContinuousPipeline(Pipeline):

    def __init__(self, brain_data_reader, stimuli_encoder, mapper, pipeline_name, save_dir="processed_data/"):
        super(ContinuousPipeline, self).__init__(brain_data_reader, stimuli_encoder, mapper, pipeline_name, save_dir)

        # Delay is measured in TRs, a delay of 2 means that we align stimulus 1 with scan 3.
        self.delay = 2
        self.voxel_selection = "on_train_ev"
        # # Example: [(detrend, {'t_r': 2.0})]
        self.voxel_preprocessings = []
        self.metrics = {"R2": r2_score_complex, "Explained Variance": explained_variance_complex,
                         "Pearson_squared": pearson_jain_complex}
        self.data = {}
        # # Options for voxel selection are: "on_train_r", "on_train_ev"random", "by_roi", "stable" and "none"
        self.voxel_selection ="on_train_ev"
        self.roi = []
        self.save_dir = save_dir
        self.voxel_to_region_mappings = {}


    def process(self, experiment_name):

        # Reading data
        self.prepare_data()
        # Iterate over subjects
        print("Subjects: " + str(self.data.keys()))
        if self.subject_ids == None:
            self.subject_ids =  list(self.data.keys())
        for subject_id in self.subject_ids:
            print("Start processing for SUBJECT: " + str(subject_id))
            subject_data = self.data[subject_id]
            self.evaluate_crossvalidation(subject_id, subject_data, experiment_name)

    def get_all_predictive_voxels(self, experiment_name):

        # Reading data
        self.prepare_data()
        # Iterate over subjects
        print("Subjects: " + str(self.data.keys()))
        if self.subject_ids == None:
            self.subject_ids = list(self.data.keys())
        for subject_id in self.subject_ids:
            print("Start processing for SUBJECT: " + str(subject_id))
            subject_data = self.data[subject_id]
            all_scans = []
            all_embeddings = []

            # Iterate through blocks and collect all scans and embeddings
            for key, value in self.data[subject_id].items():
                scans = value[0]

                embeddings = value[1]
                all_scans.extend(scans)
                all_embeddings.extend(embeddings)
            predictive_voxels = self.get_predictive_voxels(explained_variance_score,all_embeddings, all_scans ,0,  experiment_name + "_" + str(subject_id))


    def runRSA(self, experiment_name):

        # Reading data
        self.prepare_data()
        # Iterate over subjects

        if self.subject_ids == None:
            self.subject_ids =  list(self.data.keys())
        print("Subjects: " + str(self.subject_ids))
        subject_scans =[]
        subject_embeddings = []
        print(self.data.keys())
        for subject_id in self.subject_ids:
            print("Start processing for SUBJECT: " + str(subject_id))
            all_scans = []
            all_embeddings = []

            # Iterate through blocks and collect all scans and embeddings
            for key,value in self.data[subject_id].items():
                scans = value[0]
                embeddings = value[1]
                all_scans.extend(scans)
                all_embeddings.extend(embeddings)

            # Preprocess voxels
            all_scans = preprocess_voxels(self.voxel_preprocessings, np.asarray(all_scans))
            # TODO might want to add voxel selection here
            # Collect data from subjects
            subject_scans.extend(all_scans)
            subject_embeddings.extend(all_embeddings)
            # Calculate similarity between scans and embeddings
            print(datetime.datetime.now().time())

            print(np.asarray(all_scans).shape , np.asarray(all_embeddings).shape)
            x, C = get_dists([all_scans , all_embeddings])
            print(datetime.datetime.now().time())
            rdm = compute_distance_over_dists(x, C)
            print(datetime.datetime.now().time())
            print(rdm)
            score = rdm[0][1]
            print(score)
        print(datetime.datetime.now().time())
        print(np.asarray(subject_scans).shape, np.asarray(subject_embeddings).shape)
        x_all, C_all = get_dists([subject_scans, subject_embeddings])
        print(datetime.datetime.now().time())
        rdm = compute_distance_over_dists(x_all, C_all)
        scan_labels = ["Scan_"+str(id) for id in self.subject_ids ]
        embedding_labels = ["Scan_" + str(id) for id in self.subject_ids]
        plot_rdm(x_all, C_all, [scan_labels, embedding_labels])
        print(np.mean(rdm))


    def evaluate_crossvalidation(self, subject_id, subject_blocks, experiment_name):
        collected_results = {}
        collected_matches = {}
        num_blocks = len(subject_blocks.keys())
        if num_blocks >= 2:
            testkeys = subject_blocks.keys()
            all_blocks = subject_blocks
        else:
            testkeys, all_blocks = self.split_data(subject_blocks)

        fold = 1
        for testkey in testkeys:
            logging.info("Starting fold " + str(fold))
            train_data, train_targets, test_data, test_targets = self.prepare_fold(testkey, all_blocks)
            logging.info("Select voxels on the training data")
            selected_voxels = self.select_interesting_voxels(self.voxel_selection, train_data, train_targets, experiment_name, 500)

            # Reduce the scans and the predictions to the interesting voxels
            train_targets = reduce_voxels(train_targets, selected_voxels)
            test_targets = reduce_voxels(test_targets, selected_voxels)


            logging.info('Training completed. Training loss: ')
            self.mapper.train(train_data, train_targets)

            test_predictions = self.mapper.map(inputs=test_data, targets=test_targets)["predictions"]

            current_results = evaluate_fold(self.metrics, test_predictions, test_targets)

            print("Results for fold " + str(fold))
            print(str(current_results["R2"][1]))
            collected_results = update_results(current_results, collected_results)

            # pairwise_matches = pairwise_accuracy_randomized(test_predictions, test_targets, 1)
            # for key, value in pairwise_matches.items():
            #     print(key, value / float(len(test_predictions)))
            # collected_matches = add_to_collected_results(pairwise_matches, collected_matches)
            fold += 1
        print("Results for all folds")

        evaluation_path = self.save_dir + self.pipeline_name + "/evaluation_" + str(
            subject_id) +  experiment_name +"_standard_cv.txt"
        logging.info("Writing evaluation to " + evaluation_path)
        evaluation_util.save_evaluation(evaluation_path, experiment_name, subject_id, collected_results)

        pairwise_evaluation_file = self.save_dir + self.pipeline_name + "/evaluation_" + str(subject_id) +  experiment_name +"_2x2.txt"
        logging.info("Writing evaluation to " + pairwise_evaluation_file)
        save_pairwise_evaluation(pairwise_evaluation_file, self.pipeline_name, subject_id, collected_matches,
                                      len(train_targets) + len(test_targets))
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

        train_scans = preprocess_voxels(self.voxel_preprocessings, train_scans)
        test_scans = preprocess_voxels(self.voxel_preprocessings, test_scans)

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

    def prepare_data(self):
        datafile = self.save_dir + self.pipeline_name + "/aligned_data.pickle"
        print(self.pipeline_name)
        print(datafile)
        if os.path.isfile(datafile):
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
                    logging.info("Get sentence embeddings: "+ str(len(sentences)))
                    print(sentences[:5])
                    # Note: we assumed that all subjects had been presented with the same stimuli,
                    # but there is a tiny difference in the Harry dataset leading to a different number of sentences.
                    # That's why we get the embeddings separately for each subject
                    # Todo: check what's the difference exactly, probably a + sign or so.
                    sentence_embeddings = self.stimuli_encoder.get_sentence_embeddings(
                        self.pipeline_name + "/embeddings/" + str(subject)+"_" + str(block.block_id) + "_", sentences)
                    logging.info("Sentence embeddings obtained: " + str(len(sentence_embeddings)))
                    scans = [event.scan for event in block.scan_events]

                    stimulus_pointers = [event.stimulus_pointers for event in block.scan_events]
                    stimulus_embeddings = self.extract_stimulus_embeddings(sentence_embeddings, stimulus_pointers)
                    if not len(scans) == len(stimulus_embeddings):
                        raise ValueError("Lengths disagree, scans: " + str(len(scans)) + " embeddings: " + str(
                            len(stimulus_embeddings)))
                    logging.info("Align data with delay")
                    self.data[subject][block.block_id] = self.align_representations(scans, stimulus_embeddings,
                                                                                    self.delay)
                    logging.info("Data is aligned")

                    os.makedirs(os.path.dirname(datafile), exist_ok=True)
                    with open(datafile, 'wb') as handle:
                        pickle.dump(self.data, handle)


    def extract_stimulus_embeddings(self, sentence_embeddings, stimulus_pointers):
        stimulus_embeddings = []
        print(np.asarray(sentence_embeddings).shape)
        for stimulus_pointer in stimulus_pointers:
            if len(stimulus_pointer) > 0:
                token_embeddings = []
                for (sentence_id, token_id) in stimulus_pointer:
                    print(sentence_id, token_id)
                    print(len(sentence_embeddings[sentence_id]))
                    token_embeddings.append(sentence_embeddings[sentence_id][token_id])

                if len(token_embeddings) > 1:
                    # Instead of taking the mean here, other combinations like concatenation or sum could also work.
                    stimulus_embeddings.append(np.mean(token_embeddings, axis=0))
                else:
                    stimulus_embeddings.append(token_embeddings[0])
            else:
                # I am returning the empty embedding if we do not have a stimulus. There might be smarter ways.
                stimulus_embeddings.append([])
        print(np.asarray(stimulus_embeddings).shape)
        return stimulus_embeddings

    def align_representations(self, scans, embeddings, delay):
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

        for i in range(0, len(trial_scans)):
            if (i + delay) < len(embeddings) and not len(embeddings[i + delay]) == 0:
                embedding = embeddings[i + delay]
                aligned_scans.append(trial_scans[i])
                aligned_embeddings.append(embedding)

        if not len(aligned_embeddings) == len(aligned_scans):
            raise ValueError(
                "Embedding length: " + str(len(aligned_embeddings)) + " Scan length: " + str(len(aligned_scans)))

        return aligned_scans, aligned_embeddings
