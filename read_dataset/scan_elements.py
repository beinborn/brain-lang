# The ScanEvent class should be a general object for all kinds of brain data
# Currently, we only use fMRI, but it could be extended.
# The required fields are a current guess and are likely to be modified later on.
class ScanEvent:

    subject_id = ""

    # Experimental block or run
    block = 1

    # All tokens that have been presented between the previous and the current scan
    stimulus = []

    # All complete sentences that have been presented during the block up to the previous scan:
    sentences = []

    # The sentence we are currently in:
    current_sentence = ""

    timestamp = 0.0
    scan = []

    # If available: a mapping from the nth-voxel to the corresponding brain region
    voxel_to_region_mapping = {}

class ScanEvent2:

    subject_id = ""

    # Stimulus is a list of tuples (sentence_id, token_id)
    stimulus = [(0,0)]

    timestamp = 0.0

    # Array of voxel activations
    scan = []



# One experimental block or run
class Block:
    block_id = 1

    subject_id = ""

    # List of sentences presented during the block, each sentence is a list of tokens.
    # Tokens are stored as presented to the participants, no preprocessing in the reader.
    sentences = [["Tokens", "in", "the", " first", "sentence."], ["And", "tokens", "in", "sentence", "number", "two."]]

    # List of scan events occurring in this block
    scan_events = []

    # If available: a mapping from the nth-voxel in each scan to the corresponding brain region
    voxel_to_region_mapping = {}


# Read all data returns dictionary {Subject: List of Blocks}