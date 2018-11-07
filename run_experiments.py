"""Main file to run for training and evaluating the models.

"""
from ExplainBrain import  ExplainBrain
from read_dataset.readHarryPotterData import HarryPotterReader
from computational_model.text_encoder import TfHubElmoEncoder


if __name__ == '__main__':

  # Define how we want to read the brain data
  brain_data_reader = HarryPotterReader()

  # Define how we want to computationaly represent the stimuli
  stimuli_encoder = TfHubElmoEncoder()

  # Build the pipeline object
  explain_brain = ExplainBrain(brain_data_reader, stimuli_encoder)

  # Train and evaluate how well we can predict the brain activatiobs
  explain_brain.train_mapper()