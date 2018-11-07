import numpy as np

# Variant a: combine word embeddings into a sentence embedding
# Variant b: combine sentence embeddings into a story embedding (for Kaplan data)
# TODO: a and b probably require different methods

def combine_wordToSentence(stimulus, embedding):
    # TODO Samira: add your weighted averaging methods here

def combine_sentenceToStory(stimulus, embedding):
    if not len(stimulus) == len(embedding):
        raise ValueError("Stimulus and embedding should both be lists of same length")

    # Simple baseline, we should do something more sophisticated here like positional averaging
    concatenated_embedding = []
    for e in embedding:
        np.concatenate(concatenated_embedding, e)
    return concatenated_embedding
