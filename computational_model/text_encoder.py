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


# The three methods for getting embeddings for sentences, words, stories are very repetitive.
# They could be combined in a single method, but this way it is easier to adjust things.
class ElmoEncoder(TextEncoder):

    def __init__(self, embedding_dir, load_previous):
        super(ElmoEncoder, self).__init__(embedding_dir)
        self.load_previous = load_previous
        self.layer_id = 1
        self.only_forward = True
        if not load_previous:
            self.embedder = ElmoEmbedder()


    # Get sentence embeddings from elmo
    # make sure that tokenization is correct for ELMO, e.g. "don't" becomes "do n't"
    # They don't specify it, but they seemed to have used spacy tokenization for that

    # Can we use elmo for Dutch? Try model from here: https://github.com/HIT-SCIR/ELMoForManyLangs

    # Takes a list of sentences and returns a list of embeddings
    def get_sentence_embeddings(self, name, sentences):
        # Layer 0 are token representations which are not sensitive to context
        # Layer 1 are representations from the first bilstm
        # Layer 2 are the representations from the second bilstm
        # TODO: Usually one learns a weighted sum over the three layers. We should discuss what to use here
        embedding_file = self.embedding_dir + name + "/sentence_embeddings.pickle"
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

        single_layer_embeddings = [embedding[self.layer_id] for embedding in sentence_embeddings[:]]



        if self.only_forward:
            forward_embeddings = []
            for sentence_embedding in single_layer_embeddings:
                forward_embeddings.append([token_embedding[0:512] for token_embedding in sentence_embedding])
            return forward_embeddings
        else:
            return single_layer_embeddings

    def get_word_embeddings(self, name, words):

        embedding_file = self.embedding_dir +name + "/word_embeddings.pickle"
        if self.load_previous:
            logging.info("Loading embeddings from " + embedding_file)
            with open(embedding_file, 'rb') as handle:
                word_embeddings = pickle.load(handle)
        else:

            word_embeddings = self.embedder.embed_batch(words)

            # Save embeddings
            os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
            with open(embedding_file, 'wb') as handle:
                pickle.dump(word_embeddings, handle)


        if not len(word_embeddings) == len(words):
            raise RuntimeError("Something went wrong with the embedding")

        token_layer_embeddings = [embedding[0] for embedding in word_embeddings[:]]

    # According to the code, forward lstm and backward lstm are concatenated.
     # By using only the first half ot eh dimensions, I assume that I am using only the forward lm.
        if self.only_forward:
            forward_embeddings = []
            for sentence_embedding in token_layer_embeddings:
                forward_embeddings.extend([token_embedding[0:512] for token_embedding in sentence_embedding])

            return forward_embeddings
        else:
            return token_layer_embeddings


    def get_story_embeddings(self, name,  stories, mode = "sentence_final"):
        embedding_file = self.embedding_dir + name + "/story_embeddings.pickle"
        if self.load_previous:
            logging.info("Loading embeddings from " + embedding_file)
            with open(embedding_file, 'rb') as handle:
                story_embeddings = pickle.load(handle)
        else:
            story_embeddings = []
            for story in stories:
                sentence_embeddings = self.get_sentence_embeddings(name, story)
                story_embedding = []
                for embedding in sentence_embeddings:
                    if mode == "sentence_final":
                        story_embedding.append(embedding[-1])
                    if mode == "mean":
                        story_embedding.append(np.mean(embedding, axis =0))

                story_embeddings.append(np.mean(story_embedding, axis =0))


            # Save embeddings
            os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
            with open(embedding_file, 'wb') as handle:
                pickle.dump(story_embeddings, handle)

        return story_embeddings
