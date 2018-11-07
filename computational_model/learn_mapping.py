import numpy as np
from scipy import linalg

# Learn linear mapping between gold fmri activations and the embeddings returned by the model
def learnMapping(gold, embeddings):
    # Where do these transformations come from? Please provide a reference.
    squared = (gold.T).dot(gold)
    inversed = linalg.inv(squared)
    anotherTransformation = (inversed).dot(gold.T)

    mapping = (anotherTransformation).dot(embeddings)
    return mapping


# Apply mapping to predict fmri activation from stimulus embedding
def makePredictions(mapping, stimuli):
    predictions = []

    for stimulus in stimuli:
        mapped = mapping.T.dot(stimulus)
        prediction = mapped.ravel()
        predictions.append(prediction)

    return predictions
