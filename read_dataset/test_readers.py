from read_dataset import read_alice_data
from read_dataset.read_harry_potter_data import HarryPotterReader
from read_dataset.read_posts_data import StoryDataReader
from read_dataset.read_alice_data import AliceDataReader
from read_dataset.read_words_data import MitchellReader
import pickle
from language_preprocessing.tokenize import SpacyTokenizer
import numpy as np

# --- Here, we read the fmri datasets. ----
# The read_all method returns a list of scan events.
# The scan events can be filtered by subject, block, timestamp etc.
# Each event contains:
# A vector of voxel activations for that particular time stamp
# The stimulus presented at the time stamp.
# Stimulus = Words presented between current and previous scan.
# Note that the voxel activations indicate the response to earlier stimuli due to the hemodynamic delay.
# We also keep track of the sentences seen so far and the current sentence.

# ---- Mitchell DATA -----
# Make sure to get the data at http://www.cs.cmu.edu/~fmri/science2008/data.html
# Adjust the dir!

# print("\n\nMitchell Data")
# mitchell_reader = MitchellReader(data_dir="/Users/lisa/Corpora/mitchell/")
# subject_id = 1
# mitchell_data = mitchell_reader.read_all_events(subject_ids=[subject_id])
#
# for block in mitchell_data[subject_id][0:5]:
#     sentences = block.sentences
#     scans = [event.scan for event in block.scan_events]
#     print(len(scans[0]))
#     stimuli = [event.stimulus_pointers for event in block.scan_events]
#     timestamps = [event.timestamp for event in block.scan_events]
#
#     print("\n\nBLOCK: " + str(block.block_id))
#     print("Number of scans: " + str(len(scans)))
#     print("Number of stimuli: " + str(len(stimuli)))
#     print("Number of timestamps: " + str(len(timestamps)))
#     print("Stimulus: \n" + str(stimuli[0]))
#     print("Word: \n" + str(sentences[0]))

# ---- ALICE DATA -----
# Make sure to get the data at https://drive.google.com/file/d/0By_8Ci8eoDI4Q3NwUEFPRExIeG8/view
# # Adjust the dir!
# alice_dir = "/Users/lisa/Corpora/alice_data/"
# roi_size = 10
#
# print("\n\nAlice in Wonderland Data")
# alice_reader = AliceDataReader(data_dir="/Users/lisa/Corpora/alice_data/")
# #
# alice_data = alice_reader.read_all_events(subject_ids = [18])
#
# for block in alice_data[18]:
#     sentences = block.sentences
#     scans = [event.scan for event in block.scan_events]
#     stimuli = [event.stimulus_pointers for event in block.scan_events]
#     timestamps = [event.timestamp for event in block.scan_events]
#     pointers = [event.stimulus_pointers for event in block.scan_events]
#
#     print("\n\nBLOCK: " + str(block.block_id))
#     print("Number of scans: " + str(len(scans)))
#     print("Number of stimuli: " + str(len(stimuli)))
#     print("Number of timestamps: " + str(len(timestamps)))
#     print("Last pointer: " + str(pointers[-1]))
#     print("Example stimuli 100-120 = Lists of (sentence_id, token_id): \n" + str(stimuli[100:120]))
#     print("Example sentences 1-3: \n" + str(sentences[0:3]))
#     print(sentences)
#     print(pointers)




# ---- NBD DATA ----
# Get the data at: https://osf.io/utpdy/
# The NBD data is very big, start with a subset.
# # Make sure to change the dir:
# nbd_dir = "/Users/lisa/Corpora/NBD/"
# #
# print("\n\nDutch Narrative Data")
# nbd_data = read_NBD_Data.read_all(nbd_dir)
# print("Number of scans: " + str(len(nbd_data)))
# print("Subjects: " + str({event.subject_id for event in nbd_data}))
# print("Runs: " + str({event.block for event in nbd_data}))
# print("Examples: ")
# for i in range(0, 10):
#     print(vars(nbd_data[i]))


#---- HARRY POTTER DATA -----
#Get the data at: http://www.cs.cmu.edu/afs/cs/project/theo-73/www/plosone/
#Make sure to change the dir!
#
# #READ
print("\n\nHarry Potter Data")
harry_reader = HarryPotterReader(data_dir="/Users/lisa/Corpora/HarryPotter/")

harry_data = harry_reader.read_all_events()
#
# tokenizer = SpacyTokenizer()
# tokenizer_fn = SpacyTokenizer.tokenize
for subject in harry_data.keys():
    for block in harry_data[subject]:
        sentences = block.sentences
        scans = [event.scan for event in block.scan_events]
        print(len(scans[0]))
        stimuli = [event.stimulus_pointers for event in block.scan_events]
        timestamps = [event.timestamp for event in block.scan_events]

        print("\n\nBLOCK: " + str(block.block_id))
        print("Number of scans: " + str(len(scans)))
        print("Number of stimuli: " + str(len(stimuli)))
        print("Number of timestamps: " + str(len(timestamps)))
        print("Example stimuli 15-20 = Lists of (sentence_id, token_id): \n" + str(stimuli[14:20]))
        print("Example sentences 1-3: \n" + str(sentences[0:3]))





# ---- KAPLAN DATA -----
# This dataset is described in Dehghani et al. 2017: https://onlinelibrary.wiley.com/doi/epdf/10.1002/hbm.23814
# I received it from Jonas Kaplan, but I am not allowed to share it.

# # NOTE: the Kaplan data is different because we only have a single averaged scan for the whole story
# print("\n\nKaplan Data")
# kaplan_reader = StoryDataReader(data_dir="/Users/lisa/Corpora/Kaplan_data/")
# subject_id = 0
# kaplan_data = kaplan_reader.read_all_events([subject_id], language="english")
# sum = 0
# for block in kaplan_data[subject_id]:
#     # These are all already sorted, so I think you don't even need timesteps.
#     sentences = block.sentences
#     scans = [event.scan for event in block.scan_events]
#     stimuli = [event.stimulus_pointers for event in block.scan_events]
#     timestamps = [event.timestamp for event in block.scan_events]
#
    #print("\n\nBLOCK: " + str(block.block_id))
    #print("Number of scans: " + str(len(scans)))
    #print("Number of sentences in story: " + str(len(sentences)))
    #print("Number of timestamps: " + str(len(timestamps)))
    #print("Example sentences 1-3: \n" + str(sentences[0:3]))
    #print("Example stimuli 0-20 = Lists of (sentence_id, token_id): \n" + str(stimuli[0:20]))


#farsi_story_data = readKaplanData.read_all("/Users/lisa/Corpora/Kaplan_data", "farsi")
#chinese_story_data = readKaplanData.read_all("/Users/lisa/Corpora/Kaplan_data", "chinese")



