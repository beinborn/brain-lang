"""Pipeline for predicting brain activations.
"""

from encoding_pipeline.pipeline_utils import *
from encoding_pipeline.pipeline import Pipeline
from evaluation.metrics import *
from evaluation.evaluation_util import *
import numpy as np
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)


# This class runs experiments on datasets with continuous stimuli.
# Continuous = Stimuli are presented continuously and stretch over many scans. Order of stimuli and scans matters.
# Note: if a pickle with data is found in the save_dir, it is automatically used.
# Make sure to delete it or change the name of the save dir, if you want to rerun everything.
class ContinuousPipeline(Pipeline):

    def __init__(self, brain_data_reader, stimuli_encoder, mapper, pipeline_name, save_dir="processed_data/"):
        super(ContinuousPipeline, self).__init__(brain_data_reader, stimuli_encoder, mapper, pipeline_name, save_dir)

        self.data = {}
        self.save_dir = save_dir
        # Delay is measured in TRs, a delay of 2 means that we align stimulus 1 with scan 3.
        self.delay = 2
        self.metrics = {"R2": r2_score_complex, "Explained Variance": explained_variance_complex,
                        "Pearson_squared": pearson_jain_complex}

        # Example: [(detrend, {'t_r': 2.0})]
        self.voxel_preprocessings = []

        # Options for voxel selection are: "on_train_r", "on_train_ev"random", "by_roi", "stable" and "none"
        self.voxel_selection = "none"
        self.roi = []

        self.voxel_to_region_mappings = {}

    # This method runs standard cross-validation over all subjects.
    # Both evaluation procedures (pairwise and voxelwise) are applied.
    def process(self, experiment_name):

        # Reading data
        self.prepare_data()
        # Iterate over subjects
        print("Subjects: " + str(self.data.keys()))
        if self.subject_ids == None:
            self.subject_ids = list(self.data.keys())

        for subject_id in self.subject_ids:
            print("Start processing for SUBJECT: " + str(subject_id))
            subject_data = self.data[subject_id]
            self.evaluate_crossvalidation(subject_id, subject_data, experiment_name)

    # This method runs representational similarity analysis for each subject.
    # It does not require a mapping model.
    def runRSA(self, experiment_name):

        # Reading data
        self.prepare_data()
        # Iterate over subjects

        if self.subject_ids == None:
            self.subject_ids = list(self.data.keys())
        print("Subjects: " + str(self.subject_ids))
        subject_scans = []
        subject_embeddings = []
        print(self.data.keys())
        scores = {}
        for subject_id in self.subject_ids:
            print("Start processing for SUBJECT: " + str(subject_id))
            all_scans = []
            all_embeddings = []

            # Iterate through blocks and collect all scans and embeddings
            for key, value in self.data[subject_id].items():
                scans = value[0]
                embeddings = value[1]
                all_scans.extend(scans)
                all_embeddings.extend(embeddings)

            # Preprocess voxels
            all_scans = preprocess_voxels(self.voxel_preprocessings, np.asarray(all_scans))

            # Not yet implemented:  might want to add voxel selection here

            # Collect data from subjects
            subject_scans.append(all_scans)
            subject_embeddings.append(all_embeddings)

            # Calculate distance between stimuli
            x, C = get_dists([all_scans, all_embeddings])

            # Calculate correlation of distances between scans and embeddings.
            spearman, pearson, kullback = compute_distance_over_dists(x, C)

            score = (spearman[0][1], pearson[0][1], kullback[0][1])
            scores[subject_id] = score

            print("Spearman, Pearson, Kullback: ")
            print(score)

        # Collect scores for each subject and calculate mean
        all_scores = scores.values()
        mean_spearman = np.mean(np.asarray([score[0] for score in all_scores]))
        mean_pearson = np.mean(np.asarray([score[1] for score in all_scores]))
        mean_kullback = np.mean(np.asarray([score[2] for score in all_scores]))

        rsa_file = self.save_dir + self.pipeline_name + "/" + experiment_name + "_RSA.txt"

        # Save results
        with open(rsa_file, "w") as eval_file:
            eval_file.write("Subject\tSpearman\t,Pearson\t,Kullback\n")
            for subject in scores.keys():
                result = scores[subject]
                eval_file.write(
                    str(subject) + "\t" + str(result[0]) + "\t" + str(result[1]) + "\t" + str(result[2]) + "\n")
            eval_file.write(
                "\nAverages:" + "\t" + str(mean_spearman) + "\t" + str(mean_pearson) + "\t" + str(mean_kullback) + "\n")

    # This method runs crossvalidation over experimental blocks.
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

            # Get data
            train_data, train_targets, test_data, test_targets = self.prepare_fold(testkey, all_blocks)
            logging.info("Select voxels on the training data")

            # Select voxels
            selected_voxels = self.select_interesting_voxels(self.voxel_selection, train_data, train_targets,
                                                             experiment_name, 500)

            # Reduce the scans and the predictions to the interesting voxels
            train_targets = reduce_voxels(train_targets, selected_voxels)
            test_targets = reduce_voxels(test_targets, selected_voxels)

            # Train mapper
            logging.info('Training completed. Training loss: ')
            self.mapper.train(train_data, train_targets)

            # Get predictions
            test_predictions = self.mapper.map(inputs=test_data, targets=test_targets)["predictions"]

            # Evaluate fold
            current_results = evaluate_fold(self.metrics, test_predictions, test_targets)
            print("Results for fold " + str(fold))
            collected_results = update_results(current_results, collected_results)

            # The pairwise evaluation procedure takes longer.
            # For rapid pilot experiments, you might want to comment this part out.
            pairwise_matches = pairwise_accuracy_randomized(test_predictions, test_targets, 1)
            for key, value in pairwise_matches.items():
                print(key, value / float(len(test_predictions)))
            collected_matches = add_to_collected_results(pairwise_matches, collected_matches)

            fold += 1

        print("Results for all folds")

        evaluation_path = self.save_dir + self.pipeline_name + "/evaluation_" + str(
            subject_id) + experiment_name + "_standard_cv.txt"
        logging.info("Writing evaluation to " + evaluation_path)
        evaluation_util.save_evaluation(evaluation_path, experiment_name, subject_id, collected_results)

        # If you do not want to do pairwise evaluation, you can comment this out.
        pairwise_evaluation_file = self.save_dir + self.pipeline_name + "/evaluation_" + str(
            subject_id) + experiment_name + "_2x2.txt"
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

    # This method reads in the data, aligns it and save it.
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

                    # Note: we assumed that all subjects had been presented with the same stimuli,
                    # but there is a tiny difference in the Harry dataset leading to a different number of sentences.
                    # That's why we get the embeddings separately for each subject
                    # See the notes on sentence tokenization in read_dataset.read_harry_potter_data
                    sentence_embeddings = self.stimuli_encoder.get_sentence_embeddings(
                        self.pipeline_name + "/embeddings/" + str(subject) + "_" + str(block.block_id) + "_", sentences)
                    logging.info("Sentence embeddings obtained: " + str(len(sentence_embeddings)))

                    scans = [event.scan for event in block.scan_events]
                    stimulus_pointers = [event.stimulus_pointers for event in block.scan_events]
                    stimulus_embeddings = self.extract_stimulus_embeddings(sentence_embeddings, stimulus_pointers)

                    if not len(scans) == len(stimulus_embeddings):
                        raise ValueError("Lengths disagree, scans: " + str(len(scans)) + " embeddings: " + str(
                            len(stimulus_embeddings)))

                    # Align scans and embeddings
                    logging.info("Align data with delay")
                    self.data[subject][block.block_id] = self.align_representations(scans, stimulus_embeddings,
                                                                                    self.delay)
                    logging.info("Data is aligned")

                    # Save data
                    os.makedirs(os.path.dirname(datafile), exist_ok=True)
                    with open(datafile, 'wb') as handle:
                        pickle.dump(self.data, handle)

    # Aligning scans and embeddings is a tricky modelling part.
    # We simply aligned each stimulus with the scan that was taken two timesteps (4 seconds in Harry) later.
    # More sophisticated hemodynamic response models might work better.
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

        # I am simply ignoring the initial scans with empty stimuli so far, there might be a better solution.
        embeddings = embeddings[i:]

        # Apply the delay
        for i in range(0, len(trial_scans)):
            if (i + delay) < len(embeddings) and not len(embeddings[i + delay]) == 0:
                embedding = embeddings[i + delay]
                aligned_scans.append(trial_scans[i])
                aligned_embeddings.append(embedding)

        if not len(aligned_embeddings) == len(aligned_scans):
            raise ValueError(
                "Embedding length: " + str(len(aligned_embeddings)) + " Scan length: " + str(len(aligned_scans)))

        return aligned_scans, aligned_embeddings

    # This method extracts the embeddings for a token sequence from the corresponding sentence embedding.
    # This only works if a sentence is represented as a list of tokens as in Elmo.
    # For other sentence embeddings, you would have to query the encoder with fragments instead or find another solution.
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


    # This method determines the voxels with the best prediction results (above threshold) when testing on test_block
    def get_all_predictive_voxels(self, experiment_name, threshold = 0.3, test_block = 4):

        # Reading data
        self.prepare_data()

        # Make sure that voxel selection is none
        self.voxel_selection = "none"

        # Iterate over subjects
        logging.info("Get predictive vocels for subjects: " + str(self.data.keys()))
        if self.subject_ids == None:
            self.subject_ids = list(self.data.keys())

        for subject_id in self.subject_ids:
            logging.info("Start processing for SUBJECT: " + str(subject_id))
            train_data =[]
            train_targets = []
            all_predictive_voxels = []

            for key,value in self.data[subject_id].items():
                print("In here!")

                scans = value[0]
                embeddings = value[1]

                # Use one block for testing. Only makes sense when there are enough blocks.
                if key == test_block:
                    test_data = embeddings
                    test_targets = scans
                else:
                    train_data.extend(embeddings)
                    train_targets.extend(scans)

            predictive_voxels = self.get_predictive_voxels(explained_variance_score, train_data, train_targets, test_data, test_targets, threshold,
                                                               experiment_name + "_" + str(subject_id))
            all_predictive_voxels.append(predictive_voxels)
            print(predictive_voxels)
        return all_predictive_voxels
