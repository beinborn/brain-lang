from read_dataset2.readHarryPotterData import HarryPotterReader
from computational_model import combine_representations, mapping, load_representations
from language_preprocessing import tokenize
from evaluation import evaluate
from voxel_preprocessing import preprocess_voxels as pv
import spacy
import numpy as np
import pickle

# READ
print("\n\nHarry Potter Data")
harry_reader = HarryPotterReader(data_dir="/Users/lisa/Corpora/HarryPotter/")
subject_id = "1"
harry_data = harry_reader.read_all([subject_id])

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
    

    #
# TOKENIZE: This takes way too long!
# harry_data = tokenize.tokenize_all(harry_data, spacy.load('en_core_web_sm'))
#
# Tokenization takes too long, so we save an interim format to save time in later runs
# with open('harrydata_subj1.pickle', 'wb') as handle:
#      pickle.dump(harry_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('harrydata_subj1.pickle', 'rb') as handle:
#     harry_data = pickle.load(handle)
# block1 = [e for e in harry_data if e.block == 1]
# block2 = [e for e in harry_data if e.block == 2]
# block3 = [e for e in harry_data if e.block == 3]
# block4 = [e for e in harry_data if (e.subject_id == "1" and e.block == 4)]
#
# scan_events = [block1, block2, block3, block4]
#
# # The last scan of each block contains all sentences seen in this block.
# sentences = [block1[-1].sentences, block2[-1].sentences, block3[-1].sentences, block4[-1].sentences]
#
# # Collect scans
# block1_scans = [event.scan for event in block1]
# block2_scans = [event.scan for event in block2]
# block3_scans = [event.scan for event in block3]
# block4_scans = [event.scan for event in block4]
#
#
#
# # Embedding takes long, so we save an interim format to save time in debugging
# # embeddings = [load_representations.elmo_embed(sents) for sents in sentences]
# # with open('harrydata_embeddings.pickle', 'wb') as handle:
# #     pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('harrydata_embeddings.pickle', 'rb') as handle:
#     embeddings = pickle.load(handle)
#
# stimulus_embeddings = []
#
# # Get embeddings for each stimulus in each block
# for block in range(0, 4):
#     block_stimulus_embeddings = combine_representations.get_stimulus_embeddings(sentences[block], embeddings[block],
#                                                                                 scan_events[block])
#     stimulus_embeddings.append(block_stimulus_embeddings)
#
# # Prepare train and test data
# train_scans = block1_scans + block2_scans + block3_scans
# train_embeddings = stimulus_embeddings[0] + stimulus_embeddings[1] + stimulus_embeddings[2]
# test_scans = block4_scans
# test_embeddings = stimulus_embeddings[3]
#
# # Set delay to 2 timesteps, for the Harry Data this equals to 4 seconds
# delay = 2
# print("Add delay to embeddings for train: " + str(len(train_scans)) + " scans and " + str(
#     len(train_embeddings)) + " embeddings. ")
#
# train_scans, train_embeddings = combine_representations.add_delay(delay, train_scans, train_embeddings)
# print("Add delay to embeddings for test: " + str(len(test_scans)) + " scans and " + str(
#     len(test_embeddings)) + " embeddings. ")
# test_scans, test_embeddings = combine_representations.add_delay(delay, test_scans, test_embeddings)
#
# # Preprocess scans
# # TODO: THink about this! In which order? Shouldn"t we apply selection over both
#
# train_scans = pv.zscore(pv.select(pv.clean(np.array(train_scans))))
# test_scans = pv.zscore(pv.select(pv.clean((np.array(test_scans)))))
#
#
# #  learn mapping from embeddings to activations
# print("Training a mapping model for: embeddings " + str(np.array(train_embeddings).shape) + " and scans " + str(train_scans.shape))
# mapping_model = mapping.learn_mapping(np.array(train_embeddings), train_scans)
#
# #  apply mapping on test data
# predicted_scans = mapping.predict(mapping_model, np.array(test_embeddings))
#
# #  Evaluate
# print("Explained variance for block 4, subject1")
# print(evaluate.voxelwise_explained_variance(block4_scans, predicted_scans))
# print(evaluate.voxelwise_explained_variance(block4_scans, predicted_scans, "raw_values"))
