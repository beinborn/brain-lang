from read_dataset import readAliceData
from read_dataset import readHarryPotterData
from read_dataset import readNBDData
from read_dataset import readKaplanData
from computational_model import load_representations
from language_preprocessing import tokenize

# --- Here, we read the fmri datasets. ----
# The read_all method returns a list of scan events.
# The scan events can be filtered by subject, block, timestamp etc.
# Each event contains:
# A vector of voxel activations for that particular time stamp
# The stimulus presented at the time stamp.
# Stimulus = Words presented between current and previous scan.
# Note that the voxel activations indicate the response to earlier stimuli due to the hemodynamic delay.
# We also keep track of the sentences seen so far and the current sentence.


# ---- ALICE DATA -----
# Make sure to get the data at https://drive.google.com/file/d/0By_8Ci8eoDI4Q3NwUEFPRExIeG8/view
# Adjust the dir!
# alice_dir = "/Users/lisa/Corpora/alice_data/"
# roi_size = 10
#
# alice_data = readAliceData.read_all(alice_dir, roi_size)
# print("\n\nAlice in Wonderland Data")
# print("Number of scans: " + str(len(alice_data)))
# print("Subjects: " + str({event.subject_id for event in alice_data}))
# print("Some examples: ")
# for i in range(0, 16):
#     print(vars(alice_data[i]))

# ---- NBD DATA ----
# Get the data at: https://osf.io/utpdy/
# The NBD data is very big, start with a subset.
# Make sure to change the dir:
# nbd_dir = "/Users/lisa/Corpora/NBD/"
# #
# print("\n\nDutch Narrative Data")
# nbd_data = readNBDData.read_all(nbd_dir)
# print("Number of scans: " + str(len(nbd_data)))
# print("Subjects: " + str({event.subject_id for event in nbd_data}))
# print("Runs: " + str({event.block for event in nbd_data}))
# print("Examples: ")
# for i in range(0, 10):
#     print(vars(nbd_data[i]))


# ---- HARRY POTTER DATA -----
# Get the data at: http://www.cs.cmu.edu/afs/cs/project/theo-73/www/plosone/
# Make sure to change the dir!

harry_dir = "/Users/lisa/Corpora/HarryPotter/"
#
# print("\n\nHarry Potter Data")
harry_data = readHarryPotterData.read_all(harry_dir)
for block in harry_data[subject_id]:
    # These are all already sorted, so I think you don't even need timesteps.
    sentences = block.sentences
    scans = [event.scan for event in block.scan_events]
    stimuli = [event.stimulus for event in block.scan_events]
    timestamps = [event.timestamp for event in block.scan_events]

    print("\n\nBLOCK: " + str(block.block_id))
    print("Number of scans: " + str(len(scans)))
    print("Number of stimuli: " + str(len(stimuli)))
    print("Number of timestamps: " + str(len(timestamps)))
    print("Example stimuli 15-20 = Lists of (sentence_id, token_id): \n" + str(stimuli[14:20]))
    print("Example sentences 1-3: \n" + str(sentences[0:3]))

test_readers.py

# ---- KAPLAN DATA -----
# This dataset is described in Dehghani et al. 2017: https://onlinelibrary.wiley.com/doi/epdf/10.1002/hbm.23814
# I received it from Jonas Kaplan, but I am not allowed to share it.

# NOTE: the Kaplan data is different because we only have a single averaged scan for the whole story
print("\n\nKaplan Data")
kaplan_reader = StoryReader(data_dir="/Users/lisa/Corpora/Kaplan_data")
subject_id = 0
kaplan_data = kaplan_reader.read_all([subject_id], language="english")

for block in kaplan_data[str(subject_id)]:
    # These are all already sorted, so I think you don't even need timesteps.
    sentences = block.sentences
    scans = [event.scan for event in block.scan_events]
    stimuli = block.sentences
    timestamps = [event.timestamp for event in block.scan_events]

    print("\n\nBLOCK: " + str(block.block_id))
    print("Number of scans: " + str(len(scans)))
    print("Number of sentences in story: " + str(len(stimuli)))
    print("Number of timestamps: " + str(len(timestamps)))
    print("Example sentences 1-3: \n" + str(sentences[0:3]))

# farsi_story_data = readKaplanData.read_all("/Users/lisa/Corpora/Kaplan_data", "farsi")
# chinese_story_data = readKaplanData.read_all("/Users/lisa/Corpora/Kaplan_data", "chinese")

# story_embeddings =
