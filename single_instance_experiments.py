"""Main file to run for training and evaluating the models.

"""
from Single_Instance_Pipeline import SingleInstancePipeline
from read_dataset.readMitchellData import MitchellReader
from read_dataset.readKaplanData import StoryDataReader
from computational_model.text_encoder import ElmoEncoder
from computational_model.random_encoder import RandomEncoder
from mapping_models.sk_mapper import SkMapper
import logging

load_previous = False

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    save_dir = "/Users/lisa/Experiments/fmri/single_instance/"

    # mitchell_dir = "/Users/lisa/Corpora/mitchell/"
    # mitchell_reader = MitchellReader(data_dir=mitchell_dir)
    #
    stimuli_encoder = ElmoEncoder(save_dir, load_previous)
    mapper = SkMapper(alpha=10.0)
    # # Basic = select 500 voxels based on explained variance, no preprocessing, text2brain, Kaplan_embeddings: sentence_final,
    # # random_encoder = RandomEncoder(save_dir, "Kaplan_RandomEmbedder", load_previous)
    # mitchell_pipeline = SingleInstancePipeline(mitchell_reader, stimuli_encoder, mapper, save_dir=save_dir)
    #
    # mitchell_pipeline.subject_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # mitchell_pipeline.task_is_text2brain = False
    # # mitchell_pipeline.process("Mitchell_Basic")
    #
    # # mitchell_pipeline.process("Mitchell_RandomEmbedder")
    # mitchell_pipeline.process("Mitchell_Basic_Brain2Text")

    #
    kaplan_dir = "/Users/lisa/Corpora/Kaplan_data/"
    kaplan_reader = StoryDataReader(data_dir=kaplan_dir)
    kaplan_pipeline = SingleInstancePipeline(kaplan_reader, stimuli_encoder, mapper, save_dir=save_dir)
    subject_ids = list(range(6, 30))
    # # Had already processed subject1
    kaplan_pipeline.subject_ids = subject_ids

    kaplan_pipeline.process("Kaplan_Basic")

# Parameters:
# Task
# Preprocessing:
# 1) spatial smoothing, yes/no
# 2) average over selected rois, yes/no HOW DO WE DETERMINE THE ROIs?

# Voxel selection:
# select voxels by variance yes/no
# select voxels by ROI -> only if not a preprocessing step
# search-light analysis


# Encoder:
# only forward: yes/no

# Always: ignore constant voxels in evaluation, evaluation metrics: ev, 2x2, rsa

# Parameters:
# Task
# Preprocessing:
# 1) average over selected rois, yes/no HOW DO WE DETERMINE THE ROIs?

# Voxel selection:
# select voxels by variance yes/no
# select voxels by ROI -> only if not a preprocessing step
# search-light analysis
