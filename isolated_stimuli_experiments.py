"""Main file to run for training and evaluating the models.

"""
from encoding_pipeline.IsolatedPipeline import SingleInstancePipeline
from read_dataset.read_words_data import MitchellReader
from read_dataset.read_posts_data import StoryDataReader
from language_models.text_encoder import ElmoEncoder
from language_models.random_encoder import RandomEncoder
from mapping_models.sk_mapper import SkMapper
import logging

load_previous = False
mitchell_dir = "/Users/lisa/Corpora/mitchell/"
kaplan_dir = "/Users/lisa/Corpora/Kaplan_data/"
save_dir = "/Users/lisa/Experiments/fmri/single_instance/"

#Use this to set up Mitchell and Kaplan experiments
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)



    mitchell_reader = MitchellReader(data_dir=mitchell_dir)
    kaplan_reader = StoryDataReader(data_dir=kaplan_dir)
    mapper = SkMapper(alpha=10.0)
    #
    stimuli_encoder = ElmoEncoder(save_dir, load_previous)
    random_encoder = RandomEncoder(save_dir, load_previous)

    mitchell_pipeline = SingleInstancePipeline(mitchell_reader, stimuli_encoder, mapper, save_dir=save_dir)
    # TODO set subect ids to all!
    mitchell_pipeline.subject_ids = [1]
    mitchell_pipeline.process("Mitchell_Basic")



    kaplan_pipeline = SingleInstancePipeline(kaplan_reader, stimuli_encoder, mapper, save_dir=save_dir + "Kaplan")
    kaplan_pipeline.process("Kaplan_Basic")