"""Main file to run for training and evaluating the models.

"""
from Single_Instance_Pipeline import Pipeline
from read_dataset.readMitchellData import MitchellReader
from computational_model.text_encoder import ElmoEncoder
from mapping_models.sk_mapper import SkMapper
import logging
load_previous = True
data_dir = "/Users/lisa/Corpora/mitchell/"
save_dir = "/Users/lisa/Experiments/fmri/Mitchell/"

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Define how we want to read the brain data
    mitchell_reader = MitchellReader(data_dir=data_dir)

    # Define how we want to computationaly represent the stimuli
    stimuli_encoder = ElmoEncoder(save_dir + "/embeddings/", load_previous)
    stimuli_encoder.layer_id = 0
    # Set the mapping model
    mapper = SkMapper(alpha=1.0)

    # Build the pipeline object
    experiment = Pipeline(mitchell_reader, stimuli_encoder, mapper, save_dir=save_dir)
    experiment.load_previous = load_previous
    # Train and evaluate
    experiment.process()
