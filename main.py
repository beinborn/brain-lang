from read_dataset import readAliceData
from read_dataset import readHarryPotterData
from read_dataset import readNBDData
from computationalModel import load_representations

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
alice_dir = "/Users/lisa/Corpora/alice_data/"
roi_size = 10

alice_data = readAliceData.read_all(alice_dir, roi_size)
print("\n\nAlice in Wonderland Data")
print("Number of scans: " + str(len(alice_data)))
print("Subjects: " + str({event.subject_id for event in alice_data}))
print("Some examples: ")
for i in range(0, 16):
    print(vars(alice_data[i]))

# ---- NBD DATA ----
# Get the data at: https://osf.io/utpdy/
# The NBD data is very big, start with a subset.
# Make sure to change the dir:
nbd_dir = "/Users/lisa/Corpora/NBD/"

print("\n\nDutch Narrative Data")
nbd_data = readNBDData.read_all(nbd_dir)
print("Number of scans: " + str(len(nbd_data)))
print("Subjects: " + str({event.subject_id for event in nbd_data}))
print("Runs: " + str({event.block for event in nbd_data}))
print("Examples: ")
for i in range(0, 10):
    print(vars(nbd_data[i]))


# ---- HARRY POTTER DATA -----
# Get the data at: http://www.cs.cmu.edu/afs/cs/project/theo-73/www/plosone/
# Make sure to change the dir!

harry_dir = "/Users/lisa/Corpora/HarryPotter/"

print("\n\nHarry Potter Data")
harry_data = readHarryPotterData.read_all(harry_dir)
print("Number of scans: " + str(len(harry_data)))
print("Subjects: " + str({event.subject_id for event in harry_data}))
print("Blocks: " + str({event.block for event in harry_data}))
print("Example: ")
print(vars(harry_data[18]))


# Check if loading representations works for one example
subject_events = [e for e in harry_data if (e.subject_id == "1" and e.block ==1)]
sentences = subject_events[-1].sentences
print("Loading elmo embeddings for: " + str(sentences))
sentence_embeddings = load_representations.elmo_embed(sentences)

