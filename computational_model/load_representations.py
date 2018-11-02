# Input: a list of sentences
# Sentence: Each sentence is a list of strings, each string is a token
# Output: a list of sentence representations
# For elmo, a sentence representation is a list of token representations

from allennlp.commands.elmo import ElmoEmbedder


# Get sentence embeddings from elmo
# TODO: make sure that tokenization is correct for ELMO, e.g. "don't" becomes "do n't" --> use allennlp word splitter? https://github.com/allenai/allennlp/blob/master/allennlp/data/tokenizers/word_splitter.py
# Can we use elmo for Dutch? Try model from here: https://github.com/HIT-SCIR/ELMoForManyLangs
def elmo_embed(sentences):
    elmo = ElmoEmbedder()

    # Layer 0 are token representations which are not sensitive to context
    # Layer 1 are representations from the forward lstm
    # Layer 2 are the representations from the backward lstm
    # Usually one learns a weighted sum over the three layers. We should discuss how to do this!
    return elmo.embed_batch(sentences)
