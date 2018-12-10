import numpy as np
import pickle
import os
import logging

# This class generates random embeddings.
# Every word is assigned a random (but fixed) vector of the requested dimensionality.
# Sentence embeddings are a combination of token embeddings.
# Story embeddings are averaged over the averages of the sentence embeddings.
# Note: If pickle files exists in the embedding dir, the encoder automatically reads them.
# Make sure to delete them, if you want new embeddings.

class RandomEncoder(object):
    def __init__(self, embedding_dir):
        self.embedding_dir = embedding_dir
        self.dimensions = 512
        self.dict = {}

        # Set the seed to make experiments reproducible.
        self.seed = 5
        np.random.seed(self.seed)

    def get_word_embeddings(self, name, words):
        dict_file = self.embedding_dir + name + "/random_word_dict.pickle"
        logging.info("Get embedding for words: " + str(words))
        logging.info("Number of words: " + str(len(words)))

        if os.path.isfile(dict_file):
            with open(dict_file, 'rb') as handle:
                self.dict = pickle.load(handle)

        embeddings = []
        for word in words:
            logging.info("Get embedding for words: " + str(word))
            word = word[0]
            if word in self.dict.keys():
                embeddings.append(self.dict[word])
            else:
                embedding = np.random.rand(self.dimensions)
                self.dict[word] = embedding
                embeddings.append(embedding)
        # Save embeddings
        os.makedirs(os.path.dirname(dict_file), exist_ok=True)

        with open(dict_file, 'wb') as handle2:
            pickle.dump(self.dict, handle2)
        logging.info("Number of embeddings: " + str(len(embeddings)))

        return embeddings


    def get_sentence_embeddings(self, name, sentences):
        embedding_file = self.embedding_dir + name + "/sentence_embeddings.pickle"
        dict_file = self.embedding_dir + name + "/random_word_dict.pickle"
        if os.path.isfile(embedding_file):
            logging.info("Loading embeddings from " + embedding_file)
            with open(embedding_file, 'rb') as handle:
                sentence_embeddings = pickle.load(handle)
        else:
            sentence_embeddings = []
            i = 0
            for sentence in sentences:
                word_embeddings = self.get_word_embeddings(name, sentence)
                print("Embedding of sentence")
                print(np.asarray(word_embeddings).shape)
                sentence_embeddings.append(word_embeddings)
                print("Embedding of all sentences in story")
                print(np.asarray(sentence_embeddings).shape)
                i += 1

            # Save embeddings
            os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
            with open(embedding_file, 'wb') as handle:
                pickle.dump(sentence_embeddings, handle)
            with open(dict_file, 'wb') as handle2:
                pickle.dump(self.dict, handle2)
        return sentence_embeddings


    def get_story_embeddings(self, name, stories):
        embedding_file = self.embedding_dir + name + "/story_embeddings.pickle"
        dict_file = self.embedding_dir + name + "/random_word_dict.pickle"
        if os.path.isfile(embedding_file):
            logging.info("Loading embeddings from " + embedding_file)
            with open(embedding_file, 'rb') as handle:
                story_embeddings = pickle.load(handle)
        else:
            story_embeddings = []
            i = 0
            for story in stories:
                sentence_embeddings = self.get_sentence_embeddings(name + "_" + str(i), story)
                sentence_means = []
                for sentence in sentence_embeddings:
                    sentence_means.append(np.mean(np.asarray(sentence), axis = 0))
                story_embedding = np.mean(np.asarray(sentence_means), axis = 0)

                # We take the mean over the sentences because
                # fMRI images in the Kaplan data are also averaged over the sentences.
                story_embeddings.append(story_embedding)
                i += 1

            # Save embeddings
            os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
            with open(embedding_file, 'wb') as handle:
                pickle.dump(story_embeddings, handle)
            with open(dict_file, 'wb') as handle2:
                pickle.dump(self.dict, handle2)
        return story_embeddings
