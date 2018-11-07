import sklearn as sk
from mapping_models.BasicMapper import BasicMapper

class SkMapper(BasicMapper):
  def __init__(self, hparams, model_fn=sk.linear_model.Ridge):
    super(SkMapper, self).init(hparams)
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
