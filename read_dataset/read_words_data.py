import numpy as np
import scipy.io
from .scan_elements import Block, ScanEvent
from .read_fmri_data_abstract import FmriReader
import os.path
import pickle

# This class reads the words data from Mitchell et al., 2008
# Paper: http://www.cs.cmu.edu/~tom/pubs/science2008.pdf
# Data: http://www.cs.cmu.edu/afs/cs/project/theo-73/www/science2008/data.html
# Make sure to also check the supplementary material.

class WordsReader(FmriReader):

    def __init__(self, data_dir):
        super(WordsReader, self).__init__(data_dir)
        self.words2scans = {}
        self.current_subject = {}
        self.stable_voxels = {}
        self.number_of_stable_voxels = 500

    def read_all_events(self, subject_ids=None):
        # Collect scan events

        blocks = {}

        if subject_ids is None:
            subject_ids = np.arange(1, 10)

        for subject_id in subject_ids:
            blocks_for_subject = []
            self.words2scans[subject_id]= {}
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

            # Mitchell et al, 2008: "A representative fMRI image for each stimulus was created by computing the mean
            # fMRI response over its six presentations, and the mean of all 60 of these representative images was
            # then subtracted from each."
            words2meanscans = {}
            for word, word_scans in self.words2scans[subject_id].items():
                # Get the mean over the six presentations
                mean_scan = np.mean(word_scans, axis =0)
                words2meanscans[word] = mean_scan

            # Get the mean of all 60 representative images
            all_mean = np.mean(np.asarray(list(words2meanscans.values())))

            # Subtract the all_mean from each representative image.
            for word, scan in words2meanscans.items():
                normalized_scan = scan - all_mean
                scan_event = ScanEvent(subject_id, [(0, 0)], n, normalized_scan)
                block = Block(subject_id, n, [[word]], [scan_event], mapping)
                blocks_for_subject.append(block)
                n+=1
            blocks[subject_id] = blocks_for_subject

        return blocks

    def get_voxel_to_region_mapping(self, subject_id):
        dirname = os.path.dirname(__file__)
        roi_file = os.path.join(dirname, "additional_data/mitchell_roi_mapping/roi_P" + str(subject_id) + ".txt")
        roi_mapping = {}
        voxel_index = 0

        with open(roi_file,"r") as roi_data:
            for line in roi_data:
                roi_mapping[voxel_index] = line.strip()
                voxel_index +=1

        return roi_mapping

    def get_voxel_to_xyz_mapping(self, subject_id):
        dirname = os.path.dirname(__file__)
        xyz_file = os.path.join(dirname, "additional_data/mitchell_xyz_mapping/xyz_P" + str(subject_id) + ".txt")
        xyz_mapping = {}
        voxel_index = 0
        with open(xyz_file, "r") as coordinates:
            for line in coordinates:
                xyz_mapping[voxel_index] = tuple([float(x) for x in (line.strip().split("\t"))])
                voxel_index += 1
        return xyz_mapping

    def get_stable_voxels_for_fold(self, subject_id, test_stimuli):

        # Order of the two test words does not matter
        key1 = test_stimuli[0] + "_" + test_stimuli[1]
        key2 = test_stimuli[1] + "_" + test_stimuli[0]

        # We set the subject id, so that we do not have to load the dictionary every time.
        # It might cause some unexpected side effects, though, if you rely on the subject_ids somewhere else.
        if not self.current_subject == subject_id:
            self.current_subject = subject_id

            # Read dictionary, key: testword1_testword2 value: list of ids for stable voxels
            dirname = os.path.dirname(__file__)
            voxel_file = os.path.join(dirname, "additional_data/mitchell_stable_voxels/stable_voxels_" + str(
                subject_id) + "all.pickle")
            with open(voxel_file, 'rb') as handle:
                self.stable_voxels = pickle.load(handle)


        if key1 in self.stable_voxels.keys():
            return self.stable_voxels[key1]

        elif key2 in self.stable_voxels.keys():
            return self.stable_voxels[key2]
        else:
            raise ValueError("Stable voxels are not available for pair: " + key1+ ", " + key2)

