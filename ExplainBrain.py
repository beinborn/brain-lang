"""Pipeline for predicting brain activations.
"""

from evaluation.metrics import  mean_explain_variance
class ExplainBrain(object):
  def __init__(self, brain_data_reader, stimuli_encoder):
    self.brain_data_reader = brain_data_reader
    self.stimuli_encoder = stimuli_encoder

  def load_brain_experiment(self):
    """Load stimili and brain measurements.

    :return:
    """
    all_events = self.brain_data_reader.read_all_events()
    blocks, time_steps, brain_activations, stimuli = self.decompose_scan_events(all_events)

    return blocks, time_steps, brain_activations, stimuli

  def encode_stimuli(self, stimuli):
    return self.stimuli_encoder.encode(stimuli)

  def metrics(self):
    {'mean_EV': mean_explain_variance}

  def eval(self, predictions, targets):
    
  def explain(self, train=True, delay=0):
    blocks, time_steps, stimuli, brain_activations = self.load_brain_experiment()
    encoded_stimuli = self.encode_stimuli(stimuli)

    train_blocks, test_blocks = get_folds(blocks)
    train_encoded_stimuli, train_brain_activations = prepare_data(train_blocks, brain_activations, encoded_stimuli, delay)
    test_encoded_stimuli, test_brain_activations = prepare_data(test_blocks, brain_activations, encoded_stimuli, delay)

    if train:
      self.mapper.train(inputs=train_encoded_stimuli,targets=train_brain_activations)

    self.eval(train_encoded_stimuli, train_brain_activations)
    self.eval(test_encoded_stimuli, test_brain_activations)

    mapper_output = self.mapper.map(inputs=train_encoded_stimuli,targets=train_brain_activations)

    for metric_name, metric_fn in self.metrics().iter():
      metric_eval = metric_fn(predictions=mapper_output, targets=brain_activations)
      print(metric_name,":",metric_eval)


