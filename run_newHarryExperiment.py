"""Main file to run for training and evaluating the models.

"""
from Single_Instance_Pipeline import Pipeline
from read_dataset.readHarryPotterData import HarryPotterReader
from computational_model.text_encoder import ElmoEncoder
from mapping_models.sk_mapper import SkMapper
import logging


data_dir = "/Users/lisa/Corpora/HarryPotter/"
save_dir = "/Users/lisa/Experiments/fmri/HarryPotter/"
load_previous = False
if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)

  # Define how we want to read the brain data
  harry_reader = HarryPotterReader(data_dir= data_dir)

  # Define how we want to computationaly represent the stimuli
  stimuli_encoder = ElmoEncoder(save_dir + "embeddings/", load_previous)

  # Set the mapping model
  mapper = SkMapper()

  # Build the pipeline object
  experiment = Pipeline(harry_reader, stimuli_encoder, mapper, save_dir =save_dir)
  experiment.load_previous = load_previous
  experiment.delay = 2
  # Train and evaluate
  experiment.process()