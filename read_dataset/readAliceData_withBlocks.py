import numpy as np
import regex as re
from os import listdir
from .scan_elements import Block, ScanEvent
from .FmriReader import FmriReader
from language_preprocessing.tokenize import SpacyTokenizer


class AliceDataReader(FmriReader):
    def __init__(self, data_dir):
        super(AliceDataReader, self).__init__(data_dir)
        # Block splits were determined by checking for end of sentences at end of scans.

        self.block_splits = [71, 132, 180, 238, 300]

    def read_all_events(self, subject_ids=None, **kwargs):
        all_blocks = {}
        self.roi_size = kwargs.get("roi", "10")
        scan_dir = self.data_dir + "alice_data_shared/" + str(self.roi_size) + "mm/"
        text_dir = self.data_dir + "alice_stim_shared/"

        stimuli = self.read_stimuli(text_dir)
        tokens = [word for (time, word) in stimuli]
        all_sentences, all_stimuli_pointers = self.get_pointers(stimuli)

        # Set subject_ids
        if subject_ids is None:
            subject_ids = listdir(scan_dir)
            for i in range(0, len(subject_ids)):
                filename = subject_ids[i]
                subject_ids[i] = int(re.search(r"\d+", filename).group())

        for subject_id in subject_ids:
            if subject_id == 33:
                continue

            scans = self.read_scans("s" + str(subject_id) + "-timecourses.txt", scan_dir)

            # Initialize indeces
            scan_time = 0.0
            last_scan_time = 0.0
            scan_events = []

            scan_number = 1
            block_number = 1
            blocks = []
            block_sentences = []
            # Due to the block splits, this code has become really ugly.
            # TODO: refactor
            seen_sentences = 0
            block_split = self.block_splits[0]

            for scan in scans:

                pointers_for_scan = [pointer for (word_time, pointer) in all_stimuli_pointers if
                                     word_time < scan_time and word_time > last_scan_time]
                sentence_pointers = set([pointer[0] for pointer in pointers_for_scan])

                for sentence_pointer in sentence_pointers:

                    if len(block_sentences) == sentence_pointer - seen_sentences:
                        block_sentences.append(all_sentences[sentence_pointer])

                if seen_sentences > 0:
                    pointers_for_scan = [(pointer[0] - seen_sentences, pointer[1]) for pointer in pointers_for_scan]

                #scan_words = [block_sentences[pointer[0]][pointer[1]] for pointer in adjusted_pointers]


                scan_event = ScanEvent(subject_id, pointers_for_scan, scan_time, scan)
                last_scan_time = scan_time
                scan_time += 2
                scan_events.append(scan_event)
                scan_number += 1
                if scan_number == block_split:
                    seen_sentences += len(block_sentences)
                    blocks.append(Block(subject_id, block_number, block_sentences, scan_events,
                                        self.get_voxel_to_region_mapping()))
                    block_sentences = []
                    scan_events = []
                    block_number += 1
                    if block_number < len(self.block_splits)+1:
                        block_split = self.block_splits[block_number - 1]

            # append last block
            blocks.append(Block(subject_id, block_number, block_sentences, scan_events,
                                self.get_voxel_to_region_mapping()))
            all_blocks[subject_id] = blocks

        return all_blocks

    def read_stimuli(self, text_dir):
        xmin = 0.0

        textdata_file = text_dir + 'DownTheRabbitHoleFinal_exp120_pad_1.TextGrid'

        # --- PROCESS TEXT STIMULI --- #
        # This is a textgrid file. Processing is not very elegant, but works for this particular example.
        # The text does not contain any punctuation except for apostrophes, only words!
        # Apostrophes are separated from the previous word, maybe I should remove the whitespace for preprocessing?
        # We have 12 min of audio/text data.
        # We save the stimuli in an array because word times are already ordered
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
                        # Praat words: "sp" = speech pause, we use an empty stimulus instead
                        if word == "sp":
                            word = ""
                        stimuli.append([xmin, word.strip()])
                i += 1
        return stimuli

    def read_scans(self, subject, scan_dir):
        # --- PROCESS FMRI SCANS --- #
        # We have 361 fmri scans.
        # They have been taken every two seconds.
        # One scan consists of entries for 6 regions --> much more condensed data than Harry Potter

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
    # I strongly recommend to use the modified version of the text alice_transcription_with_punctuation.txt in this project.
    def get_pointers(self, stimuli):
        with (open("alice_transcription_with_punctuation.txt", 'r')) as textfile:
            alice_text = textfile.read()
        tokenizer = SpacyTokenizer()
        tokenized_alice_text = tokenizer.tokenize(alice_text, sentence_mode=True)
        stimuli_pointers = []
        sentence_index = 0
        token_index = 0
        stimulus_index = 0
        time = 0

        while (sentence_index < len(tokenized_alice_text)):
            # Reached end, add final punctuation

            if stimulus_index >= len(stimuli):
                for tok_index in range(token_index, len(tokenized_alice_text[sentence_index])):
                    stimuli_pointers.append((time, (sentence_index, tok_index)))
                break
            original = tokenized_alice_text[sentence_index][token_index]
            previous_time = time
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

            if word.lower() == original.lower():
                stimulus_index += 1
                stimuli_pointers.append((time, (sentence_index, token_index)))
                token_index += 1
            else:
                # Add punctuation
                if original.strip() in [".", ",", ":", "?", ";", ")", "(", "\"", " ", "!", "-", "--", "[",
                                        "]"]:
                    time = previous_time
                    stimuli_pointers.append((time, (sentence_index, token_index)))
                    token_index += 1
                # align apostrophes: e.g. transcription is "she'd" but tokens are "she" and "'d"
                elif ("\'" in word):
                    # add next token directly to pointers
                    stimuli_pointers.append((time, (sentence_index, token_index)))
                    stimuli_pointers.append((time, (sentence_index, token_index + 1)))
                    token_index += 2
                    stimulus_index += 1
                elif (len(original.strip()) == 0):
                    token_index += 1
                    time = previous_time
                elif (len(word.strip()) == 0):
                    stimulus_index += 1
                else:
                    raise RuntimeError("Alignment error with stimulus: " + word + " at time " + str(time))

                # Reached end of current sentence

            if token_index == len(tokenized_alice_text[sentence_index]):
                sentence_index += 1
                token_index = 0

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
    # The number is the number of content scans (add 12 to get the actual scan number)
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
