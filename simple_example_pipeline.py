from encoding_pipeline.isolated_pipeline import SingleInstancePipeline
from read_dataset.read_words_data import WordsReader
from language_models.elmo_encoder import ElmoEncoder
from mapping_models.ridge_regression_mapper import RegressionMapper

# This is an example pipeline using the Mitchell data.
# You need to adjust the paths
user_dir = "USERDIR/"
mitchell_dir = user_dir + "Corpora/mitchell/"
save_dir = user_dir + "Experiments/fmri/testnew/"

if __name__ == '__main__':
    # Set the components
    mitchell_reader = WordsReader(data_dir=mitchell_dir)
    mapper = RegressionMapper()
    encoder = ElmoEncoder(save_dir)

    # Set up the pipelines
    mitchell_pipeline_name = "MitchellDataset_ElmoEncoder_"
    mitchell_pipeline = SingleInstancePipeline(mitchell_reader, encoder, mapper,
                                               mitchell_pipeline_name, save_dir=save_dir)
    # Set voxel selection
    mitchell_pipeline.voxel_selection = "none"


    mitchell_pipeline.run_voxelwiseevaluation_cv("voxelwise_evaluation")
    mitchell_pipeline.runRSA("RSA")

    # The pairwise procedure takes long!
    # You can reduce the number of subjects:
    # mitchell_pipeline.subject_ids =[1]
    # mitchell_pipeline.run_pairwise_procedure("Mitchell_pairwise_noVS")
