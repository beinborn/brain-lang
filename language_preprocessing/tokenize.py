# Code from Samira:
# In this loop, the Harry Potter_dataset is tokenized. That means that "first-year" is splitted into "first", "-","year".
# Punctuation is separated from words, i.e. "afternoon," is splitted into "afternoon" and ","
# The step variable keeps track of when the words were presented:  "first", "-","year" were all presented at the same step
# Apostrophes are not splitted: "Charlie"s" remains a single word
# for block_id in np.arange(1,5):
#     new_sentence = []
#     # tokenize
#     for word,step in zip(block_texts[block_id],block_steps[block_id]):
#         splitted_word = re.split("([A-Za-z\"]+)", word.strip())
#         last_w = ""

#

import re
import spacy


def naive_word_level_tokenizer(text):
  """

  :param text: a string
  :return:
    splitted text
  """
  splitted_text = re.split("([A-Za-z\"]+)", text)
  return splitted_text


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


# I am using spacy tokenization here, because this is what seems to be used by Elmo
def spacy_tokenize(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        sentences.append([tok.text for tok in sent])
    return sentences

