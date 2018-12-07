"""Main file to run for training and evaluating the models.

"""
from encoding_pipeline.continuous_pipeline import ContinuousPipeline
from encoding_pipeline.continuous_pipeline import ContinuousPipeline
from read_dataset.read_harry_potter_data import HarryPotterReader
from read_dataset.read_alice_data import AliceDataReader
from language_models.text_encoder import ElmoEncoder
from language_models.random_encoder import RandomEncoder
from mapping_models.sk_mapper import SkMapper
from voxel_preprocessing.preprocess_voxels import *
import logging

harry_dir = "/Users/lisa/Corpora/HarryPotter/"
alice_dir = "/Users/lisa/Corpora/alice_data/"
save_dir = "/Users/lisa/Experiments/fmri/Continuous/"

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Set the readers
    harry_reader = HarryPotterReader(data_dir=harry_dir)
    alice_reader = AliceDataReader(data_dir=alice_dir)

    # Set the language model options
    # If there are already embeddings in save_dir/pipeline_name, they will be used.
    # Set pipeline name differently, if you don't want that.
    stimuli_encoder = ElmoEncoder(save_dir)
    random_encoder = RandomEncoder(save_dir)

    # Set the mapping model
    mapper = SkMapper(alpha=10.0)

    # Set up the experiments

    for encoder in [stimuli_encoder]:
        # HARRY EXPERIMENTS WITH DIFFERENT VOXEL SELECTION
        pipeline_name = "Harry" + encoder.__class__.__name__
        harry_experiment = ContinuousPipeline(harry_reader, encoder, mapper, pipeline_name, save_dir=save_dir)
        harry_experiment.voxel_preprocessings = [(detrend, {'t_r': 2.0}), (reduce_mean, {})]


        voxel_selections = [ "none" "on_train_ev" ]
        for v_selection in voxel_selections:
            harry_experiment.voxel_selection = v_selection
        #harry_experiment.subject_ids = [1]
            harry_experiment.runRSA("rsa_"+ v_selection)
        #harry_experiment.get_all_predictive_voxels("test")

    # harry_experiment.process("test" + v_selection)

    pipeline_name = "Alice" + encoder.__class__.__name__
    alice_experiment = ContinuousPipeline(alice_reader, encoder, mapper, pipeline_name, save_dir=save_dir)
    # # The alice dataset only consists of activation values for six regions
    alice_experiment.voxel_selection = "none"
    alice_experiment.voxel_preprocessings = [(reduce_mean, {})]
    alice_experiment.runRSA("rsa")
    # alice_experiment.process("standard512_reduce_mean")
