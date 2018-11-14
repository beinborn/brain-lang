import spacy
def stimuli_tokenizer(stimuli_sequence, stimuli_steps, tokenize_fn):
  """

  :param stimuli_sequence:
  :param stimuli_steps:
  :param tokenize_fn:
  :return:
    tokens, steps
  """
  tokens = []
  steps = []

  for stimuli, step in  zip(stimuli_sequence, stimuli_steps):
    tokenized_stimuli = tokenize_fn(stimuli)
    for tok in tokenized_stimuli:
      tokens.append(tok)
      steps.append(step)

  return tokens, steps


class SpacyTokenizer(object):
    def __init__(self):
      self.nlp = spacy.load('en_core_web_sm')

    def tokenize(self, text, sentence_mode=False):
      if isinstance(text, list):
        text = ' '.join(text)

      doc = self.nlp(text)
      sentences = []
      for sent in doc.sents:
        if not sentence_mode:
          sentences.extend([tok.text for tok in sent])
        else:
          sentences.append([tok.text for tok in sent])
      return sentences