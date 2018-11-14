"""Main file to run for training and evaluating the models.

"""
from ExplainBrain import ExplainBrain
from read_dataset.readHarryPotterData import HarryPotterReader
from computational_model.text_encoder import TfHubElmoEncoder
from mapping_models.sk_mapper import SkMapper
import tensorflow as tf


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_float('alpha', 0, 'alpha')
tf.flags.DEFINE_string('embedding_dir', None, 'path to the file containing the embeddings')

hparams = FLAGS
if __name__ == '__main__':

  tf.logging.set_verbosity(tf.logging.INFO)

  # Define how we want to read the brain data
  print("1. initialize brain data reader for Harry Potter data...")
  harry_reader = HarryPotterReader(data_dir="/Users/lisa/Corpora/HarryPotter/")

  # Define how we want to computationaly represent the stimuli
  print("2. initialize text encoder ...")
  stimuli_encoder = TfHubElmoEncoder(hparams)

  print("3. initialize mapper ...")
  mapper = SkMapper(hparams)

  # Build the pipeline object
  print("4. initialize Explainer...")
  explain_brain = ExplainBrain(harry_reader, stimuli_encoder, mapper)

  # Train and evaluate how well we can predict the brain activatiobs
  print("5. train and evaluate...")
  explain_brain.train_mapper()