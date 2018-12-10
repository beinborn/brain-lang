import numpy as np
import scipy.io
from scipy.stats.stats import pearsonr
import os.path
import pickle

# Mitchell et al, 2008 use this method to select voxels.
# It takes very long to compute for all possible test pairs.
# We implemented it to our best understanding according to the descriptions in their supplementary material,
# but no guarantees that this is what they used.
# We provide the results at additional_data/mitchell_stable_voxels, so that you don't have to re-run it.

#  For the other datasets, stable voxels cannot be calculated.

def select_voxels(subject_id, data_dir):
    words2scans = {}
    datafile = scipy.io.loadmat(data_dir + "data-science-P" + str(subject_id) + ".mat")
    for i in range(0, len(datafile["data"])):
        scan = datafile["data"][i][0][0]
        word = datafile["info"][0][i][2][0]
        scans = []
        if word in words2scans:
            scans = words2scans[word]
        scans.append(scan)
        words2scans[word] = scans
    save_all_stable_voxels(subject_id, data_dir, words2scans, number_of_stable_voxels=500)

def save_all_stable_voxels(subject_id, data_dir, words2scans, number_of_stable_voxels):
    print("Selecting voxels for " + str(subject_id))
    stable_voxels_per_pair = calculate_stable_voxels(subject_id, words2scans, number_of_stable_voxels)
    print("Selected voxels for pairs: " + str(len(stable_voxels_per_pair.keys())))
    with open(data_dir + "stable_voxels_" + str(subject_id) + "all.pickle",
              'wb') as handle:
        pickle.dump(stable_voxels_per_pair, handle)


def calculate_stable_voxels(subject_id, words2scans, number_of_selected_voxels):
    stable_voxel_ids = []
    all_words = [word for word in words2scans.keys()]
    # Number of voxels differs for each subject, but is the same for every stimulus
    # I am simply checking the number of voxels for the example word "window" here.
    number_of_voxels = len(words2scans["window"][0])
    number_of_trials = 6
    trial_matrix = np.zeros((number_of_voxels, number_of_trials, len(all_words)))

    for voxel_index in np.arange(number_of_voxels):
        for word_index in np.arange(len(all_words)):
            word = all_words[word_index]
            for trial_index in np.arange(number_of_trials):
                voxel_value = words2scans[word][trial_index][voxel_index]
                trial_matrix[voxel_index][trial_index][word_index] = voxel_value

    stability_matrix = np.zeros(number_of_voxels)

    # For each voxel, calculate the correlation of the activation over all words between pairs of trials

    stable_voxels_per_pair = {}
    words = list(words2scans.keys())

    for word1 in range(0, len(all_words)):
        for word2 in range(word1 + 1, len(all_words)):
            print(word1, word2)

            for voxel_index in np.arange(number_of_voxels):
                trial_correlation_pairs = []
                for trial_index1 in np.arange(number_of_trials):
                    for trial_index2 in np.arange(trial_index1, number_of_trials):
                        data1 = trial_matrix[voxel_index][trial_index1]
                        data2 = trial_matrix[voxel_index][trial_index2]

                        # exclude the data for the two test words
                        slice1_1 = data1[0:word1]
                        slice1_2 = data1[word1 + 1:word2]
                        slice1_3 = data1[word2 + 1:]
                        vector1 = []
                        for slice in [slice1_1, slice1_2, slice1_3]:
                            if len(slice) > 0:
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
            # argpartition splits the list so that all elements higher than the kth element (in our case -500)
            # will be sorted to occur after the kth element (there is no absolute sorting, though).

            key = words[word1] + "_" + words[word2]
            stable_voxel_ids = np.argpartition(stability_matrix, -number_of_selected_voxels)[
                               -number_of_selected_voxels:]

            stable_voxels_per_pair[key] = stable_voxel_ids
            print("Saving stable voxels to dict for pair: "+ str(key))
            print("Number of keys saved already: " +str(len(stable_voxels_per_pair.keys())))
    return stable_voxels_per_pair



