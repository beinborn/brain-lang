from read_dataset import read_alice_data
from read_dataset.read_harry_potter_data import HarryPotterReader
from read_dataset.read_stories_data import StoryDataReader
from read_dataset.read_alice_data import AliceDataReader
from read_dataset.read_words_data import WordsReader
from read_dataset.read_NBD_data import NBDReader

import numpy as np

# --- With the methods below, we can test if the readers for the fmri datasets work properly. ----
# Most fMRI data is provided in matlab format. If you want to add a new reader,
# we recommend to first look at the structure of the data in Matlab
# and then use a reader for data with similar structure as basis.

# The read_all method returns a dictionary.
# The keys are the ids of the subjects.
# The value for each  subject is a list of experimental blocks.
# Each block contains a list of scan events.
# Each event contains:
#   A timestamp (when the scan was taken)
#   A vector of voxel activations for that particular time stamp
#   The stimulus presented at the time stamp. Stimulus = Words presented between current and previous scan.

# Note that the voxel activations indicate the response to earlier stimuli due to the hemodynamic delay.
# --> we need to factor this into our alignment in the experiments.
# We also keep track of all sentences that have been presented in a block.
# In the Mitchell dataset, one block consists just of a single word.

# SET THE DATA DIR: 
data_dir = "/Users/lisa/Corpora/"


# ---- Mitchell DATA -----
# Make sure to get the data at http://www.cs.cmu.edu/~fmri/science2008/data.html
# Adjust the dir!

print("\n\nMitchell Data")
mitchell_reader = WordsReader(data_dir=data_dir +"mitchell/")
subject_id = 1
mitchell_data = mitchell_reader.read_all_events(subject_ids=[subject_id])
all_scans = []
for block in mitchell_data[subject_id][0:5]:
    sentences = block.sentences
    scans = [event.scan for event in block.scan_events]

    stimuli = [event.stimulus_pointers for event in block.scan_events]
    timestamps = [event.timestamp for event in block.scan_events]

    print("\n\nBLOCK: " + str(block.block_id))
    print("Number of scans: " + str(len(scans)))
    print("Number of stimuli: " + str(len(stimuli)))
    print("Number of timestamps: " + str(len(timestamps)))
    print("Stimulus: \n" + str(stimuli[0]))
    print("Word: \n" + str(sentences[0]))

# # ---- ALICE DATA -----

# # Make sure to get the data at https://sites.lsa.umich.edu/cnllab/2016/06/11/data-sharing-fmri-timecourses-story-listening/
# # # Adjust the dir!
alice_dir = data_dir + "alice_data/"
roi_size = 10

print("\n\nAlice in Wonderland Data")
alice_reader = AliceDataReader(data_dir=data_dir + "alice_data/")

alice_data = alice_reader.read_all_events(subject_ids = [18])
all_scans = []
for block in alice_data[18]:
    sentences = block.sentences
    scans = [event.scan for event in block.scan_events]
    all_scans.append(scans[0])

    stimuli = [event.stimulus_pointers for event in block.scan_events]
    timestamps = [event.timestamp for event in block.scan_events]
    pointers = [event.stimulus_pointers for event in block.scan_events]

    print("\n\nBLOCK: " + str(block.block_id))
    print("Number of scans: " + str(len(scans)))
    print("Number of stimuli: " + str(len(stimuli)))
    print("Number of timestamps: " + str(len(timestamps)))
    print("Last pointer: " + str(pointers[-1]))
    print("Example stimuli 100-120 = Lists of (sentence_id, token_id): \n" + str(stimuli[100:120]))
    print("Example sentences 1-3: \n" + str(sentences[0:3]))
    print(sentences)
    print(pointers)





# ---- HARRY POTTER DATA -----
# Get the data at: http://www.cs.cmu.edu/afs/cs/project/theo-73/www/plosone/
# Make sure to change the dir!
#

print("\n\nHarry Potter Data")
harry_reader = HarryPotterReader(data_dir=data_dir + "HarryPotter/")

harry_data = harry_reader.read_all_events(subject_ids = [1])
stimuli = []
sents = []
interesting_sentences = []
for subject in harry_data.keys():
    for block in harry_data[subject]:

        sentences = block.sentences
        scans = [event.scan for event in block.scan_events]

        print(subject, block.block_id, len(sentences), len(scans[0]))


    print()
    stimulus = [event.stimulus_pointers for event in block.scan_events]
    timestamps = [event.timestamp for event in block.scan_events]
    stimuli.append(stimulus)
    sents.append(sentences)
    print("\n\nBLOCK: " + str(block.block_id))
    print("Number of scans: " + str(len(scans)))
    print("Number of stimuli: " + str(len(stimuli)))
    print("Number of timestamps: " + str(len(timestamps)))
    print("Example stimuli 15-20 = Lists of (sentence_id, token_id): \n" + str(stimuli[14:20]))
    print("Example sentences 1-3: \n" + str(sentences[0:3]))



# ---- KAPLAN DATA -----
# This dataset is described in Dehghani et al. 2017: https://onlinelibrary.wiley.com/doi/epdf/10.1002/hbm.23814
# I received it from Jonas Kaplan, but I am not allowed to share it.

print("\n\n Stories Data")
kaplan_reader = StoryDataReader(data_dir=data_dir + "Kaplan_data/")
kaplan_data = kaplan_reader.read_all_events(subject_ids=[29],language="english")
sum = 0

for subject_id in kaplan_data.keys():
    all_scans = []
    for block in kaplan_data[subject_id]:
        # These are all already sorted, so I think you don't even need timesteps.
        sentences = block.sentences
        scans = [event.scan for event in block.scan_events]
        stimuli = [event.stimulus_pointers for event in block.scan_events]
        timestamps = [event.timestamp for event in block.scan_events]
        all_scans.append(scans[0])

print("\n\nBLOCK: " + str(block.block_id))
print("Number of scans: " + str(len(scans)))
print("Number of sentences in story: " + str(len(sentences)))
print("Number of timestamps: " + str(len(timestamps)))
print("Example sentences 1-3: \n" + str(sentences[0:3]))
print("Example stimuli 0-20 = Lists of (sentence_id, token_id): \n" + str(stimuli[0:20]))


# The Stories data is also available in Farsi and Chinese
# farsi_story_data = readKaplanData.read_all(data_dir + "Kaplan_data", "farsi")
# chinese_story_data = readKaplanData.read_all(data_dir + "Kaplan_data", "chinese")


# ---- NBD DATA (Dutch) ----
# Get the data at: https://osf.io/utpdy/
# The NBD data is very big, start with a subset.
# # Make sure to change the dir:
# nbd_dir = data_dir + "NBD/"
# print("\n\nDutch Narrative Data")
# nbd_reader = NBDReader(data_dir=nbd_dir)
# nbd_data = nbd_reader.read_all_events()
# print("Number of scans: " + str(len(nbd_data)))
# print("Subjects: " + str({event.subject_id for event in nbd_data}))
# print("Runs: " + str({event.block for event in nbd_data}))
# print("Examples: ")
# for i in range(0, 10):
#     print(vars(nbd_data[i]))