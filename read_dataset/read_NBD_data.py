from os import listdir
import nibabel as nib
import numpy as np
from .scan_elements import ScanEvent
from .read_fmri_data_abstract import FmriReader



# This method reads the Narrative Brain Dataset that was published by Lopopolo et al 2018.
# Paper: http://lrec-conf.org/workshops/lrec2018/W9/pdf/1_W9.pdf
# Data: https://osf.io/utpdy/
# It consists of fMRI data from 24 Dutch subjects who listened to three different narratives.
# Each subject went through 6 runs:
# The first three correspond to the narratives 1-3, run 4-6 correspond to reversed versions of the narratives.
# The presentation of the runs was randomized.
# One scan was taken every 0.88 seconds.
# The stimuli have been aligned tokenwise with time stamps.
# Not yet implemented: We should probably normalize the activation by the activation of the reverse runs.

# Note: we have not used this reader in our evaluation comparison.
# That is why it does not correspond to the newest version of the classes.
# We leave it here for completeness, but you'd have to slightly adjust the read-run method to get stimuli_pointers.

class NBDReader(FmriReader):
    def __init__(self, data_dir):
        super(NBDReader, self).__init__(data_dir)

    def read_all_events(self, subject_ids=None, **kwargs):

        # --- READ TEXT STIMULI----
        stimuli = {}
        for narrative_id in range(1, 4):
            stimuli[narrative_id] = self.get_text_stimuli(self.data_dir, narrative_id)

        # --- PROCESS FMRI SCANS --- #
        all_events = []
        if subject_ids == None:
            subject_ids = [file for file in listdir(self.data_dir + 'fmri/') if file.startswith('S')]
        for subject in subject_ids:
            for narrative_id in range(1, 4):
                print("Processing data for subject: " + subject + " narrative: " + str(narrative_id))
                all_events.extend(self.read_run(self.data_dir, subject, narrative_id, stimuli[narrative_id]))

        return all_events

    def read_run(self, data_dir, subject, narrative_id, stimuli):
        fmri_dir = data_dir + 'fmri/'

        # Run 1 corresponds to narrative 1
        current_dir = fmri_dir + subject + '/run' + str(narrative_id) + '/'

        # We use vswrs files, the letters correspond to preprocessing
        # “v” = motion correction, “s" = smoothing, “w" = normalisation, “r" = realign
        fmri_data = [file for file in listdir(current_dir) if
                     (file.endswith('.nii') and file.startswith('coreg_vswrs'))]

        # Each file corresponds to one scan taken every 0.88 seconds
        scan_time = 0.0
        word_index = 0
        word_time, word = stimuli[word_index]

        events = []
        sentences = []
        seen_text = ""

        for niifile in sorted(fmri_data):

            # img.header can give a lot of metadata
            scan = nib.load(current_dir + niifile)
            image_array = np.asarray(scan.dataobj)

            voxel_vector = np.ndarray.flatten(image_array)

            # Get word sequence that has been played during previous and current scan
            word_sequence = ''
            while (word_time < scan_time) and (word_index + 1 < len(stimuli)):
                if len(word) > 0:
                    if self.is_end_of_sentence(seen_text.strip()):
                        sentences.append(seen_text)
                        seen_text = word.strip() + " "

                    # Simply add word to the current sentence
                    else:
                        if len(word) > 0:
                            seen_text = seen_text + word.strip() + " "
                    word_sequence += word + " "

                word_index += 1
                word_time, word = stimuli[word_index]
            word_sequence = word_sequence.replace('  ', ' ')

            # Set values of event
            # Not yet implemented: deriving the stimulus pointers from the stimulus
            #event = ScanEvent(subject, stimulus_pointers, scan_time, voxel_vector)


#            events.append(event)
            scan_time += 0.88

        return events

    # --- PROCESS TEXT STIMULI --- #
    # The text is already aligned with presentation times in a tab-separated file and ordered sequentially.
    def get_text_stimuli(self, data_dir, narrative_id):
        text_dir = data_dir + 'text_metadata/'
        stimuli = []

        with open(text_dir + 'Narrative_' + str(narrative_id) + '_wordtiming.txt', 'r', encoding='utf-8',
                  errors="replace") as textdata:
            for line in textdata:
                word, onset, offset, duration = line.strip().split('\t')
                stimuli.append([float(onset), word])
        return stimuli

    # Sentence boundary detection for the NBD data is very simple.
    # There occur only . and ? at the end of a sentence and they never occur in the middle of a sentence.
    def is_end_of_sentence(self, seentext):
        sentence_punctuation = (".", "?")
        if seentext.endswith(sentence_punctuation):
            return True
        else:
            return False
