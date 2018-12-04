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


class ContinuousPipelineIncremental(Pipeline):

    def __init__(self, brain_data_reader, stimuli_encoder, mapper, pipeline_name, save_dir="processed_data/"):
        super(ContinuousPipelineIncremental, self).__init__(brain_data_reader, stimuli_encoder, mapper,pipeline_name, save_dir)

        # Delay is measured in TRs, a delay of 2 means that we align stimulus 1 with scan 3.
        self.delay = 2
        self.voxel_selection = "on_train_ev"
        # # Example: [(detrend, {'t_r': 2.0})]
        self.voxel_preprocessings = []
        self.metrics = {"R2": r2_score_complex, "Explained Variance": explained_variance_complex,
                         "Pearson": pearson_complex}
        self.data = {}
        # # Options for voxel selection are: "on_train_r", "on_train_ev"random", "by_roi", "stable" and "none"
        self.voxel_selection ="on_train_r"
        self.roi = []
        self.save_dir = save_dir
        self.voxel_to_region_mappings = {}


    def process(self, experiment_name):

        # Reading data
        self.prepare_data_incremental_embeddings()
        # Iterate over subjects
        for subject_id in self.data.keys():
            print("Start processing for SUBJECT: " + str(subject_id))
            subject_data = self.data[subject_id]
            self.evaluate_crossvalidation(subject_id, subject_data, experiment_name)

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

            pairwise_matches = pairwise_accuracy_randomized(test_predictions, test_targets, 1)
            for key, value in pairwise_matches.items():
                print(key, value / float(len(test_predictions)))
            collected_matches = add_to_collected_results(pairwise_matches, collected_matches)
            fold += 1
        print("Results for all folds")

        evaluation_path = self.save_dir + self.pipeline_name + "/evaluation_" + str(
            subject_id) +  experiment_name +"_standard_cv.txt"
        logging.info("Writing evaluation to " + evaluation_path)
        evaluation_util.save_evaluation(evaluation_path, self.pipeline_name, subject_id, collected_results)

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
        if os.pathisfile(datafile):
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
                        self.pipeline_name + "/" + str(block.block_id) + "_", sentences)
                    logging.info("Sentence embeddings obtained")
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

    def prepare_data_incremental_embeddings(self):
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
                print("Number of blocks: " + str(len(blocks)))
                for block in blocks:
                    logging.info("Preparing block " + str(block.block_id))

                    self.voxel_to_region_mappings[subject] = block.voxel_to_region_mapping
                    sentences = block.sentences
                    logging.info("Get sentence embeddings")
                    stimuli = []
                    print("Number of events: " + str(len(block.scan_events)))

                    # variables used for debugging
                    empty = 0
                    pointers_length = {}
                    for event in block.scan_events:
                        pointers = np.asarray(event.stimulus_pointers)
                        l = len(pointers)
                        if l in pointers_length.keys():
                            prev = pointers_length[l]
                            pointers_length[l] = prev+1
                        else:
                            pointers_length[l]= 1

                        sentence_ids = set([pointer[0] for pointer in pointers])
                        if len(sentence_ids) == 0:
                            empty +=1
                            stimuli.append((event.timestamp, []))
                        elif len(sentence_ids) == 1:
                            sentence_id = pointers[-1][0]
                            last_token_id = pointers[-1][1]
                            stimuli.append((event.timestamp, sentences[sentence_id][0:last_token_id]))
                        else:
                            for sentence_id in sentence_ids:
                                last_token_id = np.max([pointer[1] for pointer in pointers if pointer[0] == sentence_id ])
                                stimuli.append((event.timestamp, sentences[sentence_id][0:last_token_id]))
                    print("Number of empty stimuli: "+ str(empty))
                    print("Len stimuli: " + str(len(stimuli)))
                    print("Pointers length: " + str(pointers_length))
                    stimuli_for_embedder = [stimulus[1] for stimulus in stimuli]
                    if (len(stimuli_for_embedder == 433)):
                        print("Number of different stimuli: " + len(set(stimuli_for_embedder)))
                        print("\n\n\n\n\n")
                        for s in stimuli_for_embedder:
                            print(s)
                    print("Sending stimuli to embedder: " + str(len(stimuli_for_embedder)))
                    sentence_embeddings = self.stimuli_encoder.get_sentence_embeddings(
                        self.pipeline_name + "/" + str(block.block_id) + "_", stimuli_for_embedder)
                    logging.info("Sentence embeddings obtained")

                    # Align timestamps and sentence embeddings. This is important if two sentences occur within one scan.
                    aligned_stimuli = {}
                    for i in range(0,len(sentence_embeddings)):

                        timestamp = stimuli[i][0]
                        embedding = sentence_embeddings[i]
                        if timestamp in aligned_stimuli.keys():
                            collected_embeddings = aligned_stimuli[timestamp]
                            collected_embeddings.append(embedding)
                            aligned_stimuli[timestamp] = collected_embeddings
                        else:
                            aligned_stimuli[timestamp] = [embedding]

                    scans = []
                    embeddings = []

                    for event in block.scan_events:
                        scans.append(event.scan)
                        aligned_embeddings = aligned_stimuli[event.timestamp]



                        if len(aligned_embeddings) == 1:
                            # I am taking the representation of the last token of the sentence seen so far,
                            # assuming that the model keeps a representation of the previous context.
                            # Alternative: take mean or concatenation
                            if len(aligned_embeddings[0]) == 0:
                                embedding = []
                            else:
                                embedding = aligned_embeddings[0][-1]
                        else:
                            # Take the mean over the sentence embeddings

                            sentence_embeddings = np.asarray([e[-1] for e in aligned_embeddings if len(e)>0])

                            embedding = np.mean(sentence_embeddings, axis = 0)
                        embeddings.append(embedding)

                    if not len(scans) == len(embeddings):
                        raise ValueError("Lengths disagree, scans: " + str(len(scans)) + " embeddings: " + str(
                            len(embeddings)))
                    logging.info("Align data with delay")
                    self.data[subject][block.block_id] = self.align_representations(scans, embeddings,
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
