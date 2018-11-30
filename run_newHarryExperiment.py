"""Main file to run for training and evaluating the models.

"""
from encoding_pipeline.ContinuousPipeline import ContinuousPipeline
from read_dataset.read_harry_potter_data import HarryPotterReader
from language_models.text_encoder import ElmoEncoder
from mapping_models.sk_mapper import SkMapper
import logging


data_dir = "/Users/lisa/Corpora/HarryPotter/"
save_dir = "/Users/lisa/Experiments/fmri/Continuous/HarryPotter/"
load_previous = True
if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)

  # Define how we want to read the brain data
  harry_reader = HarryPotterReader(data_dir= data_dir)

  # Define how we want to computationaly represent the stimuli
  stimuli_encoder = ElmoEncoder(save_dir, load_previous)

  # Set the mapping model
  mapper = SkMapper(alpha=10.0)

  # Build the pipeline object
  experiment = ContinuousPipeline(harry_reader, stimuli_encoder, mapper, save_dir =save_dir)
  experiment.load_previous = load_previous
  experiment.subject_ids = [1,2]
  experiment.delay = 2
  # Train and evaluate
  experiment.process("Harry_Delay2")