import tensorflow as tf
import tensorflow_hub as hub
from util.misc import pad_lists
from allennlp.commands.elmo import ElmoEmbedder
from language_preprocessing import tokenize
import numpy as np
import pickle
import os
import logging

class TextEncoder(object):
    def __init__(self, embedding_dir):
        self.embedding_dir = embedding_dir

    def get_embeddings(self, text, sequences_length):
        raise NotImplementedError()


class ElmoEncoder(TextEncoder):

    def __init__(self, embedding_dir, load_previous):
        super(ElmoEncoder, self).__init__(embedding_dir)
        if not load_previous:
            self.embedder = ElmoEmbedder()
            self.load_previous = load_previous

    # Get sentence embeddings from elmo
    # make sure that tokenization is correct for ELMO, e.g. "don't" becomes "do n't"
    # They don't specify it, but they seemed to have used spacy tokenization for that

    # Can we use elmo for Dutch? Try model from here: https://github.com/HIT-SCIR/ELMoForManyLangs

    # Takes a list of sentences and returns a list of embeddings
    def get_sentence_embeddings(self, block_id, sentences, layer_id=1, only_forward=True):

        # Layer 0 are token representations which are not sensitive to context
        # Layer 1 are representations from the first bilstm
        # Layer 2 are the representations from the second bilstm
        # TODO: Usually one learns a weighted sum over the three layers. We should discuss what to use here
        embedding_file = self.embedding_dir + str(block_id) + "/sentence_embeddings.pickle"
        if self.load_previous:
            logging.info("Loading embeddings from " + embedding_file)
            with open(embedding_file, 'rb') as handle:
                sentence_embeddings = pickle.load(handle)
        else:

            sentence_embeddings = self.embedder.embed_batch(sentences)

            # Save embeddings
            os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
            with open(embedding_file, 'wb') as handle:
                pickle.dump(sentence_embeddings, handle)

    # Shape of sentence embeddings: ( number of sentences,3, 1024)

        if not len(sentence_embeddings) == len(sentences):
            raise RuntimeError("Something went wrong with the embedding")

        single_layer_embeddings = [embedding[layer_id] for embedding in sentence_embeddings[:]]

    # TODO Is it only the forward lstm, if I use only the first half???
        if only_forward:
            forward_embeddings = []
            for sent in single_layer_embeddings:
                forward_token_embeddings = []
                for tok_embedding in sent:
                    forward_token_embeddings.append(tok_embedding[0:512])
            return forward_embeddings

        return single_layer_embeddings
