"""Main file to run for training and evaluating the models.

"""
from encoding_pipeline.ContinuousPipeline import ContinuousPipeline
from read_dataset.read_alice_data import AliceDataReader
from language_models.text_encoder import ElmoEncoder
from mapping_models.sk_mapper import SkMapper
import logging

data_dir = "/Users/lisa/Corpora/alice_data/"
save_dir = "/Users/lisa/Experiments/fmri/Continuous/Alice/"
load_previous = True
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Define how we want to read the brain data
    alice_reader = AliceDataReader(data_dir=data_dir)

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
