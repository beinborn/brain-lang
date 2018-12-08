"""Main file to run for training and evaluating the models.

"""
from encoding_pipeline.isolated_pipeline import SingleInstancePipeline
from read_dataset.read_words_data import WordsReader
from read_dataset.read_stories_data import StoryDataReader
from language_models.text_encoder import ElmoEncoder
from language_models.random_encoder import RandomEncoder
from mapping_models.sk_mapper import SkMapper
import logging

mitchell_dir = "/Users/lisa/Corpora/mitchell/"
kaplan_dir = "/Users/lisa/Corpora/Kaplan_data/"
save_dir = "/Users/lisa/Experiments/fmri/single_instance/"

# Use this to set up Mitchell and Kaplan experiments
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    mitchell_reader = WordsReader(data_dir=mitchell_dir)
    kaplan_reader = StoryDataReader(data_dir=kaplan_dir)
    mapper = SkMapper(alpha=10.0)
    #
    stimuli_encoder = ElmoEncoder(save_dir)
    random_encoder = RandomEncoder(save_dir)

    for encoder in [stimuli_encoder, random_encoder]:
        pipeline_name = "Words" + encoder.__class__.__name__
        mitchell_pipeline = SingleInstancePipeline(mitchell_reader, encoder, mapper, pipeline_name, save_dir=save_dir)
        #mitchell_pipeline.subject_ids = [1]
        mitchell_pipeline.get_all_predictive_voxels("test")

        voxel_selections = ["none"]
        for v_selection in voxel_selections:
            mitchell_pipeline.voxel_selection = v_selection
            #mitchell_pipeline.runRSA("rsa")
            #mitchell_pipeline.pairwise_procedure( v_selection + "_pairwise")
            mitchell_pipeline.standard_crossvalidation("testeval_" +v_selection)

            pipeline_name = "Posts" + encoder.__class__.__name__
            kaplan_pipeline = SingleInstancePipeline(kaplan_reader, encoder, mapper, pipeline_name, save_dir=save_dir)
            kaplan_pipeline.voxel_selection = v_selection
            #kaplan_pipeline.runRSA("rsa")
            kaplan_pipeline.standard_crossvalidation("testeval_"+ v_selection)

        #
        #
        #     # This was missing
        #     kaplan_pipeline.subject_ids = [28]
        #     kaplan_pipeline.voxel_selection ="on_train_ev"
        # # kaplan_pipeline.standard_crossvalidation(v_selection + "_crossvalidation")
        #     kaplan_pipeline.pairwise_procedure(v_selection + "_pairwise")
