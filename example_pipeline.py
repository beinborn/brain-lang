from encoding_pipeline.isolated_pipeline import SingleInstancePipeline
from read_dataset.read_words_data import WordsReader
from language_models.elmo_encoder import ElmoEncoder
from language_models.random_encoder import RandomEncoder
from mapping_models.ridge_regression_mapper import RegressionMapper
import logging

# This is an example pipeline, you need to adjust the paths
user_dir = "USER_DIR"
mitchell_dir = user_dir + "corpora/mitchell/"
save_dir = user_dir + "/Experiments/fmri/mitchell/"

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Set the components
    mitchell_reader = WordsReader(data_dir=mitchell_dir)
    mapper = RegressionMapper(alpha=10.0)
    stimuli_encoder = ElmoEncoder(save_dir)
    random_encoder = RandomEncoder(save_dir)

    # Try different language models
    for encoder in [stimuli_encoder, random_encoder]:

        # Set up the pipelines
        mitchell_pipeline_name = "Words" + encoder.__class__.__name__
        mitchell_pipeline = SingleInstancePipeline(mitchell_reader, encoder, mapper, mitchell_pipeline_name, save_dir=save_dir)
        mitchell_pipeline.voxel_selection = "none"

        mitchell_pipeline.pairwise_procedure("Mitchell_pairwise_noVS")
        mitchell_pipeline.run_standard_crossvalidation("Mitchell_CV_noVS")
        mitchell_pipeline.runRSA("Mitchell_RSA")


