"""Pipeline for predicting brain activations.
"""

from evaluation.metrics import  mean_explain_variance
from util.misc import get_folds
class ExplainBrain(object):
  def __init__(self, brain_data_reader, stimuli_encoder):
    self.brain_data_reader = brain_data_reader
    self.stimuli_encoder = stimuli_encoder
    self.blocks = None
    self.folds = None

  def load_brain_experiment(self):
    """Load stimili and brain measurements.

    :return:
    time_steps: dictonary that contains the steps for each block. dic key is block number.
    brain_activations: {block_number: {time_step: vector of brain activation}
    stimuli: {block_number: {time_step: stimuli representation}
    """
    all_events = self.brain_data_reader.read_all_events()
    blocks, time_steps, brain_activations, stimuli = self.decompose_scan_events(all_events)

    self.blocks = blocks
    return time_steps, brain_activations, stimuli

  def decompose_scan_events(self, scan_events):
    """

    :param scan_events:
    :return:
      time_steps: dictonary that contains the steps for each block. dic key is block number.
      brain_activations: {block_number: {time_step: vector of brain activation}
      stimuli: {block_number: {time_step: stimuli representation}
    """
    time_steps = {}
    brain_activations = {}
    stimuli = {}

    for event in scan_events:
      block = event.block
      if block not in brain_activations:
        brain_activations[block] = {}
        stimuli[block] = {}
        time_steps[block] = []
      time_steps[block].append(event.time_step)
      stimuli[block][event.time_step] = event.stimulus
      brain_activations[block][event.time_step] = event.scan

    return list(brain_activations.keys()), time_steps, brain_activations, stimuli


  def encode_stimuli(self, stimuli):
    """Applies the text encoder on the stimuli.

    :param stimuli:
    :return:
    """
    return self.stimuli_encoder.encode(stimuli)

  def metrics(self):
    """
    :return:
      Dictionary of {metric_name: metric_function}
    """
    {'mean_EV': mean_explain_variance}

  def eval_mapper(self, encoded_stimuli, brain_activations):
    """Evaluate the mapper based on the defined metrics.

    :param encoded_stimuli:
    :param brain_activations:
    :return:
    """
    mapper_output = self.mapper.map(inputs=encoded_stimuli,targets=brain_activations)

    for metric_name, metric_fn in self.metrics().iter():
      metric_eval = metric_fn(predictions=mapper_output, targets=brain_activations)
      print(metric_name,":",metric_eval)

  def get_blocks(self):
    """

    :return: blocks of the loaded brain data.
    """
    return self.blocks

  def get_folds(self, fold_index):
    if self.folds in None:
      self.folds = get_folds(self.blocks)

    return self.folds[fold_index]



  def train_mapper(self, delay=0, eval=True, save=True, fold_index=-1):

    # Load the brain data
    time_steps, stimuli, brain_activations = self.load_brain_experiment()

    # Encode the stimuli and get the representations from the computational model.
    encoded_stimuli = self.encode_stimuli(stimuli)

    # Get the test and training sets
    train_blocks, test_blocks = self.get_folds(fold_index)

    # Pepare the data for the mapping model (test and train sets)
    train_encoded_stimuli, train_brain_activations = self.mapper.prepare_inputs(blocks=train_blocks,
                                                                                timed_targets=brain_activations,
                                                                                timed_inputs=encoded_stimuli,
                                                                                delay=delay)
    test_encoded_stimuli, test_brain_activations = self.mapper.prepare_inputs(blocks=test_blocks,
                                                                              timed_targets=brain_activations,
                                                                              timed_inputs=encoded_stimuli,
                                                                              delay=delay)

    # Train the mapper
    self.mapper.train(inputs=train_encoded_stimuli,targets=train_brain_activations)

    # Evaluate the mapper
    if eval:
      self.eval(train_encoded_stimuli, train_brain_activations)
      self.eval(test_encoded_stimuli, test_brain_activations)




