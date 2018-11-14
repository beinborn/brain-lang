import numpy as np
import regex as re
from os import listdir
from .scan_elements import Block, ScanEvent
from .FmriReader import FmriReader
import spacy


class AliceDataReader(FmriReader):
    def __init__(self, data_dir):
        super(AliceDataReader, self).__init__(data_dir)

    def read_all_events(self, subject_ids=None, **kwargs):
        blocks = {}
        self.roi_size = kwargs.get("roi", "10")
        scan_dir = self.data_dir + "alice_data_shared/" + str(self.roi_size) + "mm/"
        text_dir = self.data_dir + "alice_stim_shared/"

        # TODO limit subject range by input parameter subject_ids and clean code
        for alice_subject in listdir(scan_dir):
            # Skip this subject because the data is corrupted

            if alice_subject == "s33-timecourses.txt":
                continue

            subject_id = int(re.search(r"\d+", alice_subject).group())
            stimuli = self.read_stimuli(text_dir)
            tokens = [word for (time, word) in stimuli]

            sentences = self.get_sentences(tokens)

            scans = self.read_scans(alice_subject, scan_dir)

            # Initialize indeces
            scan_time = 0.0
            word_index = 0
            word_time, word = stimuli[word_index]
            scan_events = []
            for scan in scans:
                stimulus_pointer = []
                while word_time < scan_time:
                    stimulus_pointer.append((0, word_index))
                    word_index += 1
                    word_time = stimuli[word_index][0]
                scan_event = ScanEvent(subject_id, stimulus_pointer, scan_time, scan)
                scan_time += 2
                scan_events.append(scan_event)
            block = Block(subject_id, 1, sentences, scan_events, self.get_voxel_to_region_mapping())
            blocks[subject_id] = [block]
        return blocks

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

    # TODO: Detect sentence boundaries and collect sentences seen so far.
    # TODO: There is no punctuation in the stimulus data, we need to get it from here: https://www.cs.cmu.edu/~rgs/alice-I.html
    # Problem: need to adjust alignment then
    # and reintroduce it
    def get_sentences(self, tokens):
        sentences = " ".join(tokens)
        sentences = [[sentences.replace("  ", " ")]]
        return sentences

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
