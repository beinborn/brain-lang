class BasicMapper(object):
  def __init__(self, hparams):
    self.hparams = hparams

  def build(self, is_train):
    raise NotImplementedError

  def map(self, inputs, targets):
    raise NotImplementedError


