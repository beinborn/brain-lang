"""Main file to run for training and evaluating the models.

"""
from encoding_pipeline.isolated_pipeline import SingleInstancePipeline
from read_dataset.read_words_data import MitchellReader
from read_dataset.read_posts_data import StoryDataReader
from language_models.text_encoder import ElmoEncoder
from language_models.random_encoder import RandomEncoder
from mapping_models.sk_mapper import SkMapper
import logging

load_previous = True
mitchell_dir = "/Users/lisa/Corpora/mitchell/"
kaplan_dir = "/Users/lisa/Corpora/Kaplan_data/"
save_dir = "/Users/lisa/Experiments/fmri/single_instance/"

#Use this to set up Mitchell and Kaplan experiments
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    mitchell_reader = MitchellReader(data_dir=mitchell_dir)
    #kaplan_reader = StoryDataReader(data_dir=kaplan_dir)
    mapper = SkMapper(alpha=10.0)
    #
    stimuli_encoder = ElmoEncoder(save_dir)
    #random_encoder = RandomEncoder(save_dir, load_previous)

    for encoder in [stimuli_encoder]:
        pipeline_name = "Words_" + encoder.__class__.__name__
        mitchell_pipeline = SingleInstancePipeline(mitchell_reader, encoder, mapper, pipeline_name,  save_dir=save_dir)

        # First pairwise
        mitchell_pipeline.voxel_selections = "on_train_r"
        mitchell_pipeline.standard_crossvalidation("crossvalidation_vsOnTrain_byR_OnlyForward")
        mitchell_pipeline.pairwise_procedure("vsOnTrain_byR_OnlyForward_pairwise")

        mitchell_pipeline.voxel_selections = "on_train_ev"
        mitchell_pipeline.standard_crossvalidation("crossvalidation_vsOnTrain_byEV_OnlyForward")
        mitchell_pipeline.pairwise_procedure("vsOnTrain_byEV_OnlyForward_pairwise")

        mitchell_pipeline.voxel_selections = "random"
        mitchell_pipeline.standard_crossvalidation("crossvalidation_vsOnTrain_byEV_OnlyForward")
        mitchell_pipeline.pairwise_procedure("vsRandom_OnlyForward_pairwise")

        # kaplan_pipeline = SingleInstancePipeline(kaplan_reader, encoder, mapper, save_dir=save_dir)
        # kaplan_pipeline.voxel_selections = "on_train"
        # kaplan_pipeline.pipeline_name = "Posts_" + encoder.__class__.__name__
        # kaplan_pipeline.pairwise_procedure("vs_byev_on_train_pairwise")

        # Then crossvalidation
        # voxel_selections = [ "none", "random"]
        # for v_selection in voxel_selections:
        #     mitchell_pipeline.voxel_selections = v_selection
        #     mitchell_pipeline.pipeline_name = "Words_" + encoder.__class__.__name__
        #     mitchell_pipeline.standard_crossvalidation("crossvalidation_vs_"+ v_selection)
        #
        # # Then croessvalidation for
        # for v_selection in ["on_train", "none", "random"]:
        #     kaplan_pipeline.voxel_selections = v_selection
        #     kaplan_pipeline.pipeline_name = "Posts_" + encoder.__class__.__name__
        #     kaplan_pipeline.standard_crossvalidation("crossvalidation_vs_" + v_selection)

