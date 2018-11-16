"""Main file to run for training and evaluating the models.

"""
from Pipeline import Pipeline
from read_dataset.readKaplanData import StoryDataReader
from computational_model.text_encoder import ElmoEncoder
from mapping_models.sk_mapper import SkMapper
from evaluation.metrics import mean_explain_variance
import logging


# TODO: decide how to represent stories!!!
data_dir = "/Users/lisa/Corpora/Kaplan_data/"
save_dir = "/Users/lisa/Experiments/fmri/Kaplan/"
load_previous = True

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)

  # Define how we want to read the brain data
  kaplan_reader = StoryDataReader(data_dir= data_dir)

  # Define how we want to computationaly represent the stimuli
  stimuli_encoder = ElmoEncoder(save_dir + "embeddings/", load_previous)

  # Set the mapping model
  mapper = SkMapper(alpha = 1.0)

  # Build the pipeline object
  experiment = Pipeline(kaplan_reader, stimuli_encoder, mapper, save_dir =save_dir)
  # Train and evaluate
  experiment.process()