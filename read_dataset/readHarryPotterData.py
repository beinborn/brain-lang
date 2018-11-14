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

    def read_all_events(self, subject_ids=None):
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
        block = Block(subject_id=subject_id, block_id=block_id)

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
            tokens = []
            words = [word for (timestamp, word) in timed_words if
                     (timestamp < scan_times[i] and timestamp >= scan_times[i - 1])]

        for i in range(scan_index_start, scan_index_end + 1):

            tokens = []
            words = [word for (timestamp, word) in timed_words if
                     (timestamp < scan_times[i] and timestamp >= scan_times[i - 1])]

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

            # Set a scan event
            scan_event = ScanEvent(subject_id=subject_id, stimulus_pointer=tokens,
                                   timestamp=scan_times[i], scan=scans[i])


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


