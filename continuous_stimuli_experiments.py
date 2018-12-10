"""Main file to run for training and evaluating the models.

"""
from encoding_pipeline.continuous_pipeline import ContinuousPipeline
from encoding_pipeline.continuous_pipeline import ContinuousPipeline
from read_dataset.read_harry_potter_data import HarryPotterReader
from read_dataset.read_alice_data import AliceDataReader
from language_models.text_encoder import ElmoEncoder
from language_models.random_encoder import RandomEncoder
from mapping_models.ridge_regression_mapper import RegressionMapper
from voxel_preprocessing.preprocess_voxels import *
import logging

# This contains the experimental code for the NAACL submission.
# Make sure to adjust the paths
user_dir = "/Users/USERNAME/"
harry_dir = user_dir + "/Corpora/HarryPotter/"
alice_dir = user_dir + "/Corpora/alice_data/"
save_dir = user_dir +"Experiments/fmri/Continuous/"

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Set the readers
    harry_reader = HarryPotterReader(data_dir=harry_dir)
    alice_reader = AliceDataReader(data_dir=alice_dir)

    # Set the language model options
    # If there are already embeddings in save_dir/pipeline_name, they will be used.
    # Set pipeline name differently, if you don't want that.
    stimuli_encoder = ElmoEncoder(save_dir)
    stimuli_encoder.only_forward = True
    random_encoder = RandomEncoder(save_dir)

    # Set the mapping model
    mapper = RegressionMapper(alpha=10.0)

    # Set up the experiments

    for encoder in [ stimuli_encoder]:

        # --- Harry experiments --
        pipeline_name = "Harry" + encoder.__class__.__name__
        harry_experiment = ContinuousPipeline(harry_reader, encoder, mapper, pipeline_name, save_dir=save_dir)
        harry_experiment.voxel_preprocessings = [(detrend, {'t_r': 2.0}), (reduce_mean, {})]

        voxel_selections = ["none", "on_train_ev" ]
        for v_selection in voxel_selections:
            harry_experiment.voxel_selection = v_selection
            harry_experiment.process("NAACL_" +v_selection)
            harry_experiment.runRSA("rsa_")

        # --- Alice experiments --
        pipeline_name = "Alice" + encoder.__class__.__name__
        alice_experiment = ContinuousPipeline(alice_reader, encoder, mapper, pipeline_name, save_dir=save_dir)
        alice_experiment.voxel_selection = "none"
        alice_experiment.voxel_preprocessings = [(reduce_mean, {})]
        alice_experiment.runRSA("rsa")

