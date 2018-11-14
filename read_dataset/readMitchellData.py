import numpy as np
import scipy.io
from .scan_elements import Block, ScanEvent
from .FmriReader import FmriReader


# This method reads the Harry Potter data that was published by Wehbe et al. 2014
# Paper: http://aclweb.org/anthology/D/D14/D14-1030.pdf
# Data: http://www.cs.cmu.edu/afs/cs/project/theo-73/www/plosone/


# It consists of fMRI data from 8 subjects who read chapter 9 of the first book of Harry Potter.
# They see one word every 0.5 seconds.
# A scan is taken every two seconds.
# The chapter was presented in four blocks of app. 12 minutes.
# Voxel size: 3 x 3 x 3

class MitchellReader(FmriReader):

    def __init__(self, data_dir):
        super(MitchellReader, self).__init__(data_dir)

    def read_all_events(self, subject_ids=None):
        # Collect scan events
        blocks = {}

        if subject_ids is None:
            subject_ids = np.arange(1, 10)

        for subject_id in subject_ids:
            blocks_for_subject = []
            datafile = scipy.io.loadmat(self.data_dir + "data-science-P" + str(subject_id) + ".mat")
            mapping = self.get_voxel_to_region_mapping(datafile["meta"]["colToCoord"])
            for i in range(0, len(datafile["data"])):
                scan = datafile["data"][i][0][0]
                word = datafile["info"][0][i][2][0]
                print(scan)
                print(word)
                scan_event = ScanEvent(subject_id, [(0, 0)], i, scan)
                block = Block(subject_id, i, [[word]], [scan_event], mapping)
                blocks_for_subject.append(block)
            blocks[subject_id] = blocks_for_subject
        return blocks

    def get_voxel_to_region_mapping(self, colToCoord):
        # I am assuming that the data is already MNI normalized, because the number of voxels is the same for all subjects.
        # But I am not completely sure.
        for voxel in range(0, len(colToCoord)):
            # TODO ask Samira or Rochelle how to get the labels, Matlab script is quite cryptic
            print(colToCoord[0][0][voxel])

        return {}
