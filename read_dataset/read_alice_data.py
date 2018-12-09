import numpy as np
import regex as re
from os import listdir
from .scan_elements import Block, ScanEvent
from .read_fmri_data_abstract import FmriReader
from language_preprocessing.tokenize import SpacyTokenizer
import os

#  This method reads the "Alice in Wonderland" data described in Brennan et al. (2016).
# Paper:https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4893969/
# Data:  https://sites.lsa.umich.edu/cnllab/2016/06/11/data-sharing-fmri-timecourses-story-listening/

class AliceDataReader(FmriReader):
    def __init__(self, data_dir):
        super(AliceDataReader, self).__init__(data_dir)
        self.block_splits = [59, 120, 168, 226, 288]
    def read_all_events(self, subject_ids=None, **kwargs):
        blocks = {}
        # There are two region of interest sizes, we just 10 mm.
        self.roi_size = kwargs.get("roi", "10")

        # Read in the data and the stimuli.
        scan_dir = self.data_dir + "alice_data_shared/" + str(self.roi_size) + "mm/"
        text_dir = self.data_dir + "alice_stim_shared/"

        stimuli = self.read_stimuli(text_dir)
        sentences, stimuli_pointers = self.get_pointers(stimuli)

        # Make sure everything was read correctly.
        print("Number of stimuli: " + str( len(stimuli)))
        print(stimuli[:9])
        print("Number of stimuli pointers: " + str(len(stimuli_pointers)))
        print(stimuli_pointers[:9])
        print("Last pointer: " + str(stimuli_pointers[-1]))
        print("Number of sentences: " + str(len(sentences)))

        # Set subject_ids
        if subject_ids is None:
            subject_ids = listdir(scan_dir)
            for i in range(0, len(subject_ids)):
                filename = subject_ids[i]
                subject_ids[i] = int(re.search(r"\d+", filename).group())

        # Skip subject 33
        for subject_id in subject_ids:
            if subject_id == 33:
                continue

            scans = self.read_scans("s" + str(subject_id) + "-timecourses.txt", scan_dir)

            # Initialize indeces
            scan_time = 0.0
            last_scan_time = 0.0
            scan_events = []


            # Note: When we align scans and stimuli, the last two sentences are missing because
            # with 362 scans and a TR of 2 seconds, we only have 724 seconds of data
            # but the stimuli extend until second 740
            # In the paper, they say that participants listened to the "first 12 minutes" of the chapter.
            # We can only assume that the rest of the story was discarded.
            # TODO: I send John Brennan a mail: waiting for a response.

            for scan in scans:
                pointers_for_scan = [pointer for (word_time, pointer) in stimuli_pointers if
                                     word_time < scan_time and word_time > last_scan_time]

                scan_event = ScanEvent(subject_id, pointers_for_scan, scan_time, scan)
                last_scan_time = scan_time
                scan_time += 2
                scan_events.append(scan_event)
            block = Block(subject_id, 1, sentences, scan_events, self.get_voxel_to_region_mapping())
            blocks[subject_id] = [block]
            print("Number of scans: " + str(len(scans)))

        return blocks

    def read_stimuli(self, text_dir):
        xmin = 0.0

        textdata_file = text_dir + 'DownTheRabbitHoleFinal_exp120_pad_1.TextGrid'

        # --- PROCESS TEXT STIMULI --- #
        # This is a textgrid file. Processing is not very elegant, but works for this particular example.
        # The text does not contain any punctuation except for apostrophes, only words!
        # Apostrophes are separated from the previous word, this might not be ideal for some language processing models.
        # We have 12 min of audio/text data.
        # We can save the stimuli in a list because word times are already ordered.
        stimuli = []

        with open(textdata_file, 'r') as textdata:
            i = 0
            for line in textdata:
                if i > 13:
                    line = line.strip()
                    # xmin should always be read BEFORE word
                    if line.startswith("xmin"):
                        xmin = float(line.split(" = ")[1])
                    if line.startswith("text"):
                        word = line.split(" = ")[1].strip("\"")
                        #  "sp" stands for speech pause, we use an empty stimulus instead.
                        if word == "sp":
                            word = ""
                        stimuli.append([xmin, word.strip()])
                i += 1
        return stimuli

    def read_scans(self, subject, scan_dir):
        # --- PROCESS FMRI SCANS --- #
        # We have 361 fmri scans.
        # They have been taken every two seconds.
        # One scan consists of entries for 6 regions --> much more condensed data than the raw voxel activations.

        with (open(scan_dir + subject, 'r')) as subjectdata:
            scans = []
            for line in subjectdata:
                # Read activation values from file
                activation_strings = line.strip().split("   ")

                # Convert string values to floats
                activations = []
                for a in activation_strings:
                    activations.append(float(a))

                scans.append(activations)
        return scans

    # Aligning the transcription (which had no punctuation) to the Gutenberg version of the text (with punctuation)
    # was an awful experience. I would not do it again.
    # The transcripts contain so many deviations from the original (e.g. missing words, typos etc)
    # which I had to fix manually.
    # I strongly recommend to use the modified version of the text alice_transcription_with_punctuation.txt in this project
    # and not change anything in this method.
    def get_pointers(self, stimuli):
        dirname = os.path.dirname(__file__)
        transcription_file = os.path.join(dirname, "additional_data/alice_transcription_with_punctuation.txt")
        with (open(transcription_file, 'r')) as textfile:
            alice_text = textfile.read()
        tokenizer = SpacyTokenizer()
        tokenized_alice_text = tokenizer.tokenize(alice_text, sentence_mode=True)
        stimuli_pointers = []
        sentence_index = 0
        token_index = 0
        stimulus_index = 0

        while (sentence_index < len(tokenized_alice_text)):
            # Reached end, add final punctuation
            if stimulus_index >= len(stimuli):
                print("Reached end: " + str(sentence_index))
                print(stimuli_pointers[-1])
                for tok_index in range(token_index, len(tokenized_alice_text[sentence_index])):
                    stimuli_pointers.append((time, (sentence_index, tok_index)))
                break
            original = tokenized_alice_text[sentence_index][token_index]
            time = stimuli[stimulus_index][0]
            word = stimuli[stimulus_index][1]

            # Correct typos in data
            if (word == "Zeland"):
                word = "Zealand"
            if (word == "happpens"):
                word = "happens"
            if time == 463.4450166426263:
                word = "through"
            if time == 477.0722496596:
                word = "knew"

            if len(word) > 0:
                if word.lower() == original.lower():
                    stimulus_index += 1
                    stimuli_pointers.append((time, (sentence_index, token_index)))
                    token_index += 1
                else:
                    # Add punctuation
                    if original.strip() in [".", ",", ":", "?", ";", ")", "(", "\"", " ", "!", "-", "--", "[",
                                            "]"] or len(original.strip()) == 0:

                        stimuli_pointers.append((time, (sentence_index, token_index)))
                        token_index += 1
                    # align apostrophes: e.g. transcription is "she'd" but tokens are "she" and "'d"
                    elif ("\'" in word):
                        # add next token directly to pointers
                        stimuli_pointers.append((time, (sentence_index, token_index)))
                        stimuli_pointers.append((time, (sentence_index, token_index + 1)))
                        token_index += 2
                        stimulus_index += 1
                    else:
                        raise RuntimeError("Alignment error with stimulus: " + word + " at time " + str(time))

                # Reached end of current sentence

                if token_index == len(tokenized_alice_text[sentence_index]):
                    sentence_index += 1
                    token_index = 0

            # Drop empty stimuli.  The reader replaced break signals with ""
            else:
                stimulus_index += 1
        return tokenized_alice_text, stimuli_pointers

    # Note that region names are not the same as for the Wehbe data!
    # The abbreviations stand for:
    # LATL: left anterior temporal lobe
    # RATL: right anterior temporal lobe
    # LPTL: left posterior temporal lobe
    # LIPL: left inferior parietal lobe
    # LPreM: left premotor
    # LIFG: left inferior frontal gyrus

    def get_voxel_to_region_mapping(self):
        return {0: "LATL", 1: "RATL", 2: "LPTL", 3: "LIPL", 4: "LPreM", 5: "LIFG", }


    # These methods can be used to split the data into n blocks
    # Potential breaks indicate sentence boundaries that coincide with scan boundaries.
    # The number is the number of content scans (add 12 [= number of resting scans] to get the actual scan number)
    def get_break_points(self, n):
        potential_breaks = [9, 19, 24, 43, 47, 59, 68, 80, 85, 87, 108, 111, 120, 130, 134, 139, 144, 145, 165, 168,
                            191, 202, 210, 226, 227, 229, 234, 251, 269, 288, 297, 301, 310, 334, 336, 338, 345]
        scans_in_fold = 351 / float(n)
        print("Scans in fold should be: " + str(scans_in_fold))
        previous_break_point = 1
        breakpoints = []
        for fold in range(1, n + 1):
            next_ideal_breakpoint = previous_break_point + scans_in_fold
            current_break_point = self.find_nearest(potential_breaks, next_ideal_breakpoint)
            print(current_break_point - previous_break_point - int(scans_in_fold))
            breakpoints.append(current_break_point)
            previous_break_point = current_break_point
        print(breakpoints)

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]