from sklearn.linear_model import *
from mapping_models.basic_mapper import BasicMapper
import numpy as np

class SkMapper(BasicMapper):
  def __init__(self, alpha = 100, model_fn=Ridge):
    super(SkMapper, self).__init__()
    self.alpha = alpha
    self.model_fn = model_fn
    self.model = None

  def build(self, is_train=True):
    """Create the model object using model_fn
    """
    self.model = self.model_fn(alpha=self.alpha)

  def map(self, inputs, targets=None):
    if self.model is None:
      self.build()
    predictions = self.model.predict(inputs)

    loss = None
    if targets is not None:
      # How is the loss computed?
      loss = self.compute_loss(predictions, targets)
    print("Loss: " + str(loss))
    return {'predictions': predictions,
            'loss': loss}

  def train(self, inputs, targets):
    if self.model is None:
      self.build()

    self.model.fit(inputs, targets)





