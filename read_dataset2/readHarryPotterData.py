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

class HarryPotterReader(FmriReader):

    def __init__(self, data_dir):
        super(HarryPotterReader, self).__init__(data_dir)

    def read_all(self, subject_ids=None):
        # Collect scan events
        blocks = {}

        if subject_ids is None:
            subject_ids = np.arange(1, 9)

        for subject_id in subject_ids:
            blocks_for_subject = []
            for block_id in range(1, 5):
                block = self.read_block(subject_id, block_id)
                block.voxel_to_region_mapping = self.get_voxel_to_region_mapping(subject_id)
                blocks_for_subject.append(block)
            blocks[subject_id] = blocks_for_subject

        return blocks

    def read_block(self, subject_id, block_id):
        # Initialize
        block = Block()
        block.subject_id = subject_id
        block.block_id = block_id

        # Data is in matlab format
        # Data structure is a dictionary with keys data, time, words, meta
        # Shapes for subject 1, block 1: data (1351,37913), time (1351,2) words (1, 5176)
        datafile = scipy.io.loadmat(self.data_dir + "subject_" + str(subject_id) + ".mat")

        # We have one scan every 2 seconds
        timedata = datafile["time"]
        scan_times = timedata[:, 0]

        # We have four blocks. One block includes approx. 12 minutes
        blocks = timedata[:, 1]

        # find first and last scan time of current block
        block_starts = np.min(scan_times[np.where(blocks == block_id)])
        block_ends = np.max(scan_times[np.where(blocks == block_id)])

        # --- PROCESS TEXT STIMULI -- #
        # Here we extract the presented words and align them with their timestamps.
        # The original data consists of weirdly nested arrays.
        timed_words = []
        presented_words = datafile["words"]

        for i in np.arange(presented_words.shape[1]):
            token = presented_words[0][i][0][0][0][0]
            timestamp = presented_words[0][i][1][0][0]
            timed_words.append((timestamp, token))

        # Get scan indices for block
        scan_index_start = np.where(scan_times == block_starts)[0][0]
        scan_index_end = np.where(scan_times == block_ends)[0][0]

        # Initialize counters
        scan_events = []
        sentences = []
        sentence_id = 0
        token_id = 0

        # Iterate through scans
        scans = datafile["data"]
        for i in range(scan_index_start, scan_index_end + 1):

            # Set a scan event
            scan_event = ScanEvent()
            scan_event.subject_id = subject_id
            scan_event.scan = scans[i]
            scan_event.timestamp = scan_times[i]

            tokens = []
            words = [word for (timestamp, word) in timed_words if
                     (timestamp < scan_times[i] and timestamp >= scan_times[i - 1])]
            #print(words)
            # Collect sentences and align stimuli
            for word in words:
                # We have not figured out what the @ should stand for and just remove it
                word = word.replace("@", "")
                if len(sentences) > 0:
                    if is_beginning_of_new_sentence(sentences[sentence_id], word):
                        sentence_id += 1
                        token_id = 0
                        sentences.append([word])
                    else:
                        sentences[sentence_id].append(word)
                        token_id += 1

                # Add first word to sentences
                else:
                    sentences = [[word]]

                tokens.append((sentence_id, token_id))

            scan_event.stimulus = tokens
            scan_events.append(scan_event)

        block.scan_events = scan_events
        block.sentences = sentences
        #print(vars(block.scan_events[20]))
        return block

    # The metadata provides very rich information.
    # Double-check description.txt in the original data.
    # Important: Each voxel in the scan has different coordinates depending on the subject!
    # Voxel 5 has the same coordinates in all scans for subject 1.
    # Voxel 5 has the same coordinates in all scans for subject 2, but they differ from the coordinates for subject 1.
    # Same with regions: Each region spans a different set of voxels depending on the subject!
    def get_voxel_to_region_mapping(self, subject_id):
        metadata = scipy.io.loadmat(self.data_dir + "subject_" + str(subject_id) + ".mat")["meta"]
        roi_of_nth_voxel = metadata[0][0][8][0]
        roi_names = metadata[0][0][10][0]
        voxel_to_region = {}
        for voxel in range(0, roi_of_nth_voxel.shape[0]):
            roi = roi_of_nth_voxel[voxel]
            voxel_to_region[voxel] = roi_names[roi][0]
        # for name in roi_names:
        #  print(name[0])
        return voxel_to_region


# This is a quite naive sentence boundary detection that only works for this dataset.
def is_beginning_of_new_sentence(sentence, newword):
    seentext = ' '.join(sentence)
    sentence_punctuation = (".", "?", "!", ".\"", "!\"", "?\"", "+")
    # I am ignoring the following exceptions, because they are unlikely to occur in fiction text:
    # "etc.", "e.g.", "cf.", "c.f.", "eg.", "al.
    exceptions = ("Mr.", "Mrs.")
    if seentext.endswith(exceptions):
        return False
    # This would not work if everything is lowercased!
    if seentext.endswith(sentence_punctuation) and not newword.islower() and newword is not ".":
        return True
    else:
        return False

# --------
# These are some lines for processing the metadata which are not needed here, but I leave them in for reference.
# Setting indices according to description.txt in the original data folder
# sub_id_index = 0
# number_of_scans_index = 1
# number_of_voxels = 2
# x_dim_index = 3
# y_dim_index = 4
# z_dim_index = 5
# colToCoord_index = 6
# coordToCol_index = 7
# ROInumToName_index = 8
# ROInumsToName_3d_index = 9
# ROINames_index = 10
# voxel_size_index = 11
# matrix_index = 12

# Extract metadata
# Number of scans is constant over all subjects: 1351
# Voxel size is also constant 3x3x3
# Number of voxels varies across subjects.
# the_subject_id = metadata[0][0][sub_id_index][0][0]
# number_of_scans = metadata[0][0][number_of_scans_index][0][0]
# number_of_voxels = metadata[0][0][number_of_voxels][0][0]
# voxel_size = metadata[0][0][voxel_size_index]

# Example: get coordinates of 5th voxel for this subject
# coordinates_of_nth_voxel = metadata[0][0][6]
# coordinates_of_nth_voxel[5]
# which_voxel_for_coordinates = metadata[0][0][7]
# get voxel number for set of coordinates
# which_voxel_for_coordinates[36,7,19]

#  These coordinates differ slightly across subjects
# x_dim = metadata[0][0][x_dim_index][0][0]
# y_dim = metadata[0][0][y_dim_index][0][0]
# z_dim = metadata[0][0][z_dim_index][0][0]

# This index provides the geometric coordinates (x,y,z) of the n-th voxel
# coords = metadata[0][0][colToCoord_index]

# I am not really sure how voxels are mapped into Regions of Interest
# column_map = metadata[0][0][coordToCol_index]
# nmi_matrix = metadata[0][0][matrix_index]
# named_areas = metadata[0][0][ROInumToName_index]
# area_names_list = metadata[0][0][ROINames_index][0]

# The Appendix.pdf gives information about the fMRI preprocessing (slice timing correction etc)
# Poldrack et al 2014: The goal of spatial normalization is to transform the brain images
# from each individual in order to reduce the variability between individuals and allow
# meaningful group analyses to be successfully performed.

# TODO ask Samira: Wehbe et al used a Gaussian kernel smoother and voxel selection, you too?

# Signal Drift: : a global signal decrease with subsequently acquired images in the scan (technological artifact)
# Detrending? Tanabe & Meyer 2002:
# Because of the inherently low signal to noise ratio (SNR) of fMRI data, removal of low frequency signal
# intensity drift is an important preprocessing step, particularly in those brain regions that weakly activate.
# Two known sources of drift are noise from the MR scanner and aliasing of physiological pulsations. However,
# the amount and direction of drift is difficult to predict, even between neighboring voxels.

# from nilearn documentation:
# Standardize signal: set to unit variance
# low_pass, high_pass: Respectively low and high cutoff frequencies, in Hertz.
# Low-pass filtering improves specificity.
# High-pass filtering should be kept small, to keep some sensitivity.
