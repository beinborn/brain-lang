import numpy as np
import scipy.io
from .scan_elements import Block, ScanEvent
from .read_fmri_data_abstract import FmriReader
from scipy.stats.stats import pearsonr
import os.path
import pickle


class MitchellReader(FmriReader):

    def __init__(self, data_dir):
        super(MitchellReader, self).__init__(data_dir)
        self.words2scans = {}
        self.stable_voxels = {}
        self.number_of_stable_voxels = 500

    def read_all_events(self, subject_ids=None):
        # Collect scan events
        blocks = {}

        if subject_ids is None:
            subject_ids = np.arange(1, 10)

        for subject_id in subject_ids:
            blocks_for_subject = []
            self.words2scans[subject_id] = {}
            datafile = scipy.io.loadmat(self.data_dir + "data-science-P" + str(subject_id) + ".mat")
            mapping = self.get_voxel_to_region_mapping(subject_id)
            for i in range(0, len(datafile["data"])):
                scan = datafile["data"][i][0][0]
                word = datafile["info"][0][i][2][0]
                scans = []
                if word in self.words2scans[subject_id]:
                    scans = self.words2scans[subject_id][word]
                scans.append(scan)
                self.words2scans[subject_id][word] = scans

            n = 1
            for word, word_scans in self.words2scans[subject_id].items():
                mean_scan = np.mean(word_scans, axis =0)
                scan_event = ScanEvent(subject_id, [(0, 0)], n, mean_scan)
                block = Block(subject_id, n, [[word]], [scan_event], mapping)
                blocks_for_subject.append(block)
                n+=1
            blocks[subject_id] = blocks_for_subject




        return blocks

    def get_voxel_to_region_mapping(self, subject_id):
        roi_file = self.data_dir + "additional_data/mitchell_roi_mapping/roi_P" + str(subject_id) + ".txt"
        #print(roi_file)
        roi_mapping = {}
        voxel_index = 0
        with open(roi_file,"r") as roi_data:
            for line in roi_data:
                roi_mapping[voxel_index] = line.strip()
                voxel_index +=1
        #print(roi_mapping)
        return roi_mapping

    def get_stable_voxels_for_fold(self, subject_id, test_stimuli):
        key = test_stimuli[0] + "_" + test_stimuli[1]
        return self.stable_voxels[key]



    def calculate_stable_voxels(self, subject_id,number_of_selected_voxels):
        stable_voxel_ids = []
        all_words = [word for word in self.words2scans[subject_id].keys()]
        # Number of voxels differs for each subject, but is the same for every stimulus
        # I am simply checking the number of voxels for the example word "window" here.
        number_of_voxels = len(self.words2scans[subject_id]["window"][0])
        number_of_trials = 6
        trial_matrix = np.zeros((number_of_voxels, number_of_trials, len(all_words)))

        for voxel_index in np.arange(number_of_voxels):
            for word_index in np.arange(len(all_words)):
                word = all_words[word_index]
                for trial_index in np.arange(number_of_trials):

                    voxel_value =  self.words2scans[subject_id][word][trial_index][voxel_index]
                    trial_matrix[voxel_index][trial_index] [word_index]= voxel_value

        stability_matrix = np.zeros(number_of_voxels)

        # For each voxel, calculate the correlation of the activation over all words between pairs of trials
        words2scans = self.words2scans[subject_id]
        stable_voxels_per_pair = {}
        words = list(words2scans.keys())

        for word1 in range(0, len(all_words)):
            for word2 in range(word1 + 1, len(all_words)):
                print(word1, word2)
                stable_voxels_per_pair = {}

                for voxel_index in np.arange(number_of_voxels):
                    trial_correlation_pairs = []
                    for trial_index1 in np.arange(number_of_trials):
                        for trial_index2 in np.arange(trial_index1, number_of_trials):
                            data1 =trial_matrix[voxel_index][trial_index1]
                            data2 = trial_matrix[voxel_index][trial_index2]

                            # exclude the data for the two test words
                            slice1_1 = data1[0:word1]
                            slice1_2 = data1[word1 + 1:word2]
                            slice1_3 = data1[word2 + 1:]
                            vector1 = []
                            for slice in [slice1_1, slice1_2, slice1_3]:
                                if len(slice)>0:
                                    vector1.extend(slice)
                            slice2_1 = data2[0:word1]
                            slice2_2 = data2[word1 + 1:word2]
                            slice2_3 = data2[word2 + 1:]
                            vector2 = []
                            for slice in [slice2_1, slice2_2, slice2_3]:
                                if len(slice) > 0:
                                    vector2.extend(slice)

                            correlation_for_pair = pearsonr(vector1, vector2)[0]
                            trial_correlation_pairs.append(correlation_for_pair)
                # Sort the voxels by mean pearson correlation and return the indices for the n voxels with the highest values
                # argpartition sorts in ascending order, so we take from the back
                # I find this expression very hard to read, but Samira tested it, so I just keep it.
                key = words[word1]+"_"+ words[word2]
                stable_voxel_ids = np.argpartition(stability_matrix, -number_of_selected_voxels)[-number_of_selected_voxels:]
                with open(self.data_dir + "stable_voxels_" + str(subject_id) + "_" + key + ".pickle",
                          'wb') as handle:
                    pickle.dump(stable_voxel_ids, handle)
                stable_voxels_per_pair[key] = stable_voxel_ids
        return stable_voxels_per_pair
    def save_all_stable_voxels(self, subject_ids):

        stable_voxels_per_subject = {}
        for subject_id in subject_ids:
            print("Selecting voxels for " + str(subject_id))
            stable_voxels_per_pair = self.calculate_stable_voxels(subject_id,
                                                          self.number_of_stable_voxels)


            if len(set(stable_voxels_per_pair.values())) == 1:
                print("The sets of selected voxels are equal for every fold.")
            else:
                print("The sets of selected voxels are different.")
            stable_voxels_per_subject[subject_id] = stable_voxels_per_pair

            with open(self.data_dir + "stable_voxels_" + str(subject_id) + "all.pickle",
                      'wb') as handle:
                pickle.dump(stable_voxels_per_pair, handle)
        with open(self.data_dir + "stable_voxels_all_subjects.pickle",
                  'wb') as handle:
            pickle.dump(stable_voxels_per_subject, handle)
