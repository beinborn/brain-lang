"""Main file to run for training and evaluating the models.

"""
from encoding_pipeline.ContinuousPipeline import ContinuousPipeline
from read_dataset.read_harry_potter_data import HarryPotterReader
from language_models.text_encoder import ElmoEncoder
from mapping_models.sk_mapper import SkMapper
import logging


harry_dir = "/Users/lisa/Corpora/HarryPotter/"

alice_dir = "/Users/lisa/Corpora/alice_data/"
save_dir = "/Users/lisa/Experiments/fmri/Continuous/"

load_previous = True
if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)

  # Set the readers
  harry_reader = HarryPotterReader(data_dir= harry_dir)
  alice_reader = AliceDataReader(data_dir=alice_dir)

  # Set the language model options
  stimuli_encoder = ElmoEncoder(save_dir, load_previous)
  random_encoder = RandomEncoder(save_dir, load_previous)

  # Set the mapping model
  mapper = SkMapper(alpha=10.0)

  # Build the pipeline objects

  for encoder in [stimuli_encoder, random_encoder]:

    alice_experiment = ContinuousPipeline(alice_reader, encoder, mapper, save_dir=save_dir)
    alice_experiment.load_previous = load_previous
    experiment.process("Alice_" + encoder.__class__.__name__)

    harry_experiment = ContinuousPipeline(harry_reader, encoder, mapper, save_dir =save_dir)
    harry_experiment.load_previous = load_previous
    experiment.process("Harry_"+ encoder.__class__.__name__)









    # Define how we want to read the brain data


    # Define how we want to computationaly represent the stimuli
    stimuli_encoder = ElmoEncoder(save_dir, load_previous)

    # Set the mapping model
    mapper = SkMapper(alpha=1.0)

    # Build the pipeline object
    experiment = ContinuousPipeline(alice_reader, stimuli_encoder, mapper, save_dir=save_dir)
    experiment.load_previous = load_previous
    experiment.delay = 2
    experiment.subject_ids = [18, 22]
    # Train and evaluate
    experiment.process("Alice_delay2")