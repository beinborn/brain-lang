from encoding_pipeline.isolated_pipeline import SingleInstancePipeline
from read_dataset.read_words_data import WordsReader
from read_dataset.read_stories_data import StoryDataReader
from language_models.text_encoder import ElmoEncoder
from language_models.random_encoder import RandomEncoder
from mapping_models.ridge_regression_mapper import RegressionMapper
import logging

# This is an example pipeline.
user_dir = "USER_DIR"
mitchell_dir = user_dir + "corpora/mitchell/"

save_dir = user_dir + "/experiments/fmri/mitchell/"


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Set the reader
    mitchell_reader = WordsReader(data_dir=mitchell_dir)


    # Set the mapping model
    mapper = RegressionMapper()

    # Set the language models
    stimuli_encoder = ElmoEncoder(save_dir)
    random_encoder = RandomEncoder(save_dir)

    # Try different language models
    for encoder in [stimuli_encoder, random_encoder]:

        # Set up the pipelines
        name = encoder.__class__.__name__
        mitchell_pipeline = SingleInstancePipeline(mitchell_reader, encoder, mapper, name, save_dir=save_dir)

        voxel_selections = ["none", ]

        for v_selection in voxel_selections:
            # mitchell_pipeline.voxel_selection = v_selection
            #
            # # Run Words experiments
            # mitchell_pipeline.run_standard_crossvalidation("crossvalidation_" + v_selection)
            # mitchell_pipeline.runRSA("rsa")
            # mitchell_pipeline.pairwise_procedure( "pairwise_" +v_selection )

            # Run Stories experiments
            # stories_pipeline.voxel_selection = v_selection
            # stories_pipeline.run_standard_crossvalidation("crossvalidation_" + v_selection)
            stories_pipeline.runRSA("rsa")
            # stories_pipeline.pairwise_procedure("pairwise_" + v_selection)


