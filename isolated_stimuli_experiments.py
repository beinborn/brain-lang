# This contains the experimental code for the Cicling 2019 submission for the isolated stimuli.

from encoding_pipeline.isolated_pipeline import SingleInstancePipeline
from read_dataset.read_words_data import WordsReader
from read_dataset.read_stories_data import StoryDataReader
from language_models.elmo_encoder import ElmoEncoder
from language_models.random_encoder import RandomEncoder
from mapping_models.ridge_regression_mapper import RegressionMapper
import logging


#  Make sure to get the Mitchell data and the Kaplan data and adjust the paths
user_dir = "USERDIR/"
mitchell_dir = user_dir + "Corpora/mitchell/"
kaplan_dir = user_dir + "Corpora/Kaplan_data/"
save_dir = user_dir + "fmriExperiments/isolated/"

# Use this to set up Mitchell and Kaplan experiments
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Set the readers
    mitchell_reader = WordsReader(data_dir=mitchell_dir)
    kaplan_reader = StoryDataReader(data_dir=kaplan_dir)

    # Set the mapping model
    mapper = RegressionMapper(alpha=10.0)

    # Set the language models
    stimuli_encoder = ElmoEncoder(save_dir)
    random_encoder = RandomEncoder(save_dir)

    # Try different language models
    for encoder in [ stimuli_encoder, random_encoder]:

        # Set up the pipelines
        mitchell_pipeline_name = "Words" + encoder.__class__.__name__
        mitchell_pipeline = SingleInstancePipeline(mitchell_reader, encoder, mapper, mitchell_pipeline_name, save_dir=save_dir)

        stories_pipeline_name = "Stories" + encoder.__class__.__name__
        stories_pipeline = SingleInstancePipeline(kaplan_reader, encoder, mapper, stories_pipeline_name, save_dir=save_dir)

        # Set voxel selection
        voxel_selections = ["none", "on_train_ev"]

        for v_selection in voxel_selections:
            mitchell_pipeline.voxel_selection = v_selection

            # Run Words experiments
            #mitchell_pipeline.run_voxelwiseevaluation_cv("crossvalidation_" + v_selection)
            #mitchell_pipeline.run_pairwise_procedure( "pairwise_" +v_selection )

            # Run Stories experiments
            stories_pipeline.voxel_selection = v_selection
            stories_pipeline.run_voxelwiseevaluation_cv("crossvalidation_" + v_selection)
            stories_pipeline.run_pairwise_procedure("pairwise_" + v_selection)

        # Run representational similarity analysis
        mitchell_pipeline.runRSA("rsa")
        stories_pipeline.runRSA("rsa")



