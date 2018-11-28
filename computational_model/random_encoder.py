import tensorflow as tf
import tensorflow_hub as hub
from util.misc import pad_lists
from allennlp.commands.elmo import ElmoEmbedder
from language_preprocessing import tokenize
import numpy as np
import pickle
import os
import logging


class RandomEncoder(object):
    def __init__(self, embedding_dir, name, load_previous):
        self.embedding_dir = embedding_dir
        self.dimensions = 512
        self.load_previous = load_previous
        #TODO: if embedding dir contains word_dict.pickle, load it
        self.dict_file = self.embedding_dir + name +"/random_word_dict.pickle"
        self.dict = {}
        if os.path.isfile(self.dict_file) :
            with open(self.dict_file, 'rb') as handle:
                self.dict = pickle.load(handle)

    def get_word_embeddings(self, name, words):
        embedding_file = self.embedding_dir + name + "/sentence_embeddings.pickle"
        if self.load_previous:
            logging.info("Loading embeddings from " + embedding_file)
            with open(embedding_file, 'rb') as handle:
                embeddings = pickle.load(handle)
        else:

            embeddings = []
            for word in words:
                word = word[0]
                if word in self.dict.keys():
                    embeddings.append(self.dict[word])
                else:
                    embedding = np.random.rand(self.dimensions)
                    self.dict[word] = embedding
                    embeddings.append(embedding)
            # Save embeddings
            os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
            with open(embedding_file, 'wb') as handle:
                pickle.dump(embeddings, handle)
            with open(self.dict_file, 'wb') as handle2:
                pickle.dump(self.dict, handle2)

        return embeddings

    def get_sentence_embeddings(self, name, sentences):
        embedding_file = self.embedding_dir + name + "/sentence_embeddings.pickle"
        if self.load_previous:
            logging.info("Loading embeddings from " + embedding_file)
            with open(embedding_file, 'rb') as handle:
                sentence_embeddings = pickle.load(handle)
        else:
            sentence_embeddings = []
            for sentence in sentences:
                word_embeddings = self.get_word_embeddings(name, sentence)
                sentence_embedding = np.mean(np.asarray(word_embeddings), axis=0)
                sentence_embeddings.append(sentence_embedding)

            # Save embeddings
            os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
            with open(embedding_file, 'wb') as handle:
                pickle.dump(sentence_embeddings, handle)
            with open(self.dict_file, 'wb') as handle2:
                pickle.dump(self.dict, handle2)
        return sentence_embeddings

    def get_story_embeddings(self, name, stories):
        embedding_file = self.embedding_dir + name + "/story_embeddings.pickle"
        if self.load_previous:
            logging.info("Loading embeddings from " + embedding_file)
            with open(embedding_file, 'rb') as handle:
                story_embeddings = pickle.load(handle)
        else:
            story_embeddings = []
            for story in stories:
                sentence_embeddings = self.get_sentence_embeddings(name, story)
                story_embedding = np.mean(sentence_embeddings, axis=0)

                story_embeddings.append(story_embedding)

            # Save embeddings
            os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
            with open(embedding_file, 'wb') as handle:
                pickle.dump(story_embeddings, handle)
            with open(self.dict_file, 'wb') as handle2:
                pickle.dump(self.dict, handle2)
        return story_embeddings
