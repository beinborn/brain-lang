import numpy as np
from scipy import linalg
import sklearn as sk

# If we learn to decode, gold are the word embeddings and train_data are the scans.
# If we learn to encode, gold are the scans and train_data are the embeddings.

def learn_mapping(gold, train_data):
    model = sk.linear_model.Ridge
    model.fit(gold, train_data)
    return model

def predict(model, test_data):
    return model.predict(test_data)


# Double-check with Rochelle

# Learn linear mapping between gold fmri activations and the embeddings returned by the model
# def learnMapping(gold, embeddings):
#     # Where do these transformations come from? Please provide a reference.
#     squared = (gold.T).dot(gold)
#     inversed = linalg.inv(squared)
#     anotherTransformation = (inversed).dot(gold.T)
#
#     mapping = (anotherTransformation).dot(embeddings)
#     return mapping
#
#
# # Apply mapping to predict fmri activation from stimulus embedding
# def makePredictions(mapping, stimuli):
#     predictions = []
#
#     for stimulus in stimuli:
#         mapped = mapping.T.dot(stimulus)
#         prediction = mapped.ravel()
#         predictions.append(prediction)
#
#     return predictions
