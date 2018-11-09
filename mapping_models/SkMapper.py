from sklearn.linear_model import *
from mapping_models.BasicMapper import BasicMapper

class SkMapper(BasicMapper):
  def __init__(self, hparams, model_fn=Ridge):
    super(SkMapper, self).__init__(hparams)
    self.alpha = hparams.alpha
    self.model_fn = model_fn
  def build(self, is_train=True):
    self.model = self.model_fn.Ridge(alpha=self.alpha)

  def map(self, inputs, targets=None):
    predictions = self.model.predict(inputs)

    loss = None
    if targets is not None:
      loss = self.compute_loss(predictions, targets)

    return {'predictions': predictions,
            'loss': loss}

  def train(self, inputs, targets):
    self.model.fit(inputs, targets)

  def prepare_inputs(self, **kwargs):
    blocks = kwargs['blocks']
    timed_targets = kwargs['timed_targets']
    timed_inputs =  kwargs['timed_inputs']
    delay = kwargs['delay']

    inputs = []
    targets = []
    for block in blocks:
      # Get current block steps
      steps = sorted(timed_targets[block].keys())
      for step in steps:
        if step+delay in timed_inputs[block]:
          inputs.append[timed_inputs[block][step+delay]]
          targets.append(timed_targets[block][step])


    return inputs, targets



