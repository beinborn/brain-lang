import spacy


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