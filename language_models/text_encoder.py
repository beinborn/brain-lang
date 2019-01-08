from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
import pickle
import os
import logging
import tensorflow as tf
import bert
import bert.modeling
import bert.tokenization
import bert.extract_features_utils as extract_features
import re

# This class retrieves embeddings from the Elmo model by Peters et al (2018).
# See https://allennlp.org/elmo for details.
# The three methods for getting embeddings for sentences, words, stories are slightly repetitive.
# They could be combined into a single method, but I found it easier to keep them separate for debugging.
# Note: If pickle files exist in the embedding dir, the encoder automatically reads them.
# Make sure to delete them, if you want new embeddings.

class TextEncoder(object):
    def __init__(self, embedding_dir):
        self.embedding_dir = embedding_dir

    def get_embeddings(self, text, sequences_length):
        raise NotImplementedError()


class ElmoEncoder(TextEncoder):

    def __init__(self, embedding_dir, load_previous=False):
        super(ElmoEncoder, self).__init__(embedding_dir)
        self.layer_id = 1
        self.only_forward = True
        self.embedder = None
        if not load_previous:
            self.embedder = ElmoEmbedder()


    # Takes a list of sentences and returns a list of embeddings
    def get_sentence_embeddings(self, name, sentences):
        # Layer 0 are token representations which are not sensitive to context
        # Layer 1 are representations from the first bilstm
        # Layer 2 are the representations from the second bilstm

        embedding_file = self.embedding_dir + name + "sentence_embeddings.pickle"
        if os.path.isfile(embedding_file):
            logging.info("Loading embeddings from " + embedding_file)
            with open(embedding_file, 'rb') as handle:
                sentence_embeddings = pickle.load(handle)
        else:

            sentence_embeddings = self.embedder.embed_batch(sentences)

            # Save embeddings
            os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
            with open(embedding_file, 'wb') as handle:
                pickle.dump(sentence_embeddings, handle)

        if not len(sentence_embeddings) == len(sentences):
            logging.info("Something went wrong with the embedding. Number of embeddings: " + str(
                len(sentence_embeddings)) + " Number of sentences: " + str(len(sentences)))

        single_layer_embeddings = [embedding[self.layer_id] for embedding in sentence_embeddings[:]]

        if self.only_forward:
            forward_embeddings = []
            for sentence_embedding in single_layer_embeddings:
                forward_embeddings.append([token_embedding[0:512] for token_embedding in sentence_embedding])
            return forward_embeddings
        else:
            return single_layer_embeddings

    def get_word_embeddings(self, name, words):

        embedding_file = self.embedding_dir + name + "word_embeddings.pickle"
        if os.path.isfile(embedding_file):
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

        # According to the elmo code, forward lstm and backward lstm are concatenated.
        # By using only the first half of the dimensions, I assume that I am using only the forward lm.
        # If you want to use the full language model you need to adjust the stimulus that you feed in.
        # If you feed in the whole sentence, the representation of token 2 will contain information from future tokens.
        #  which might not have been perceived by the subject yet.
        if self.only_forward:
            forward_embeddings = []
            for sentence_embedding in token_layer_embeddings:
                forward_embeddings.extend([token_embedding[0:512] for token_embedding in sentence_embedding])

            return forward_embeddings
        else:
            return token_layer_embeddings

    def get_story_embeddings(self, name, stories, mode="sentence_final"):
        embedding_file = self.embedding_dir + name + "story_embeddings.pickle"
        # Careful, if the file exists, I load it. Make sure to delete it, if you want to reencode.
        if os.path.isfile(embedding_file):
            logging.info("Loading embeddings from " + embedding_file)
            with open(embedding_file, 'rb') as handle:
                story_embeddings = pickle.load(handle)
        else:
            story_embeddings = []
            i = 0
            for story in stories:
                sentence_embeddings = self.get_sentence_embeddings(name + "_" + str(i), story)
                story_embedding = []
                for embedding in sentence_embeddings:
                    if mode == "sentence_final":
                        story_embedding.append(embedding[-1])
                    if mode == "mean":
                        story_embedding.append(np.mean(embedding, axis=0))
                story_embeddings.append(np.mean(story_embedding, axis=0))
                i += 1

            # Save embeddings
            os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
            with open(embedding_file, 'wb') as handle:
                pickle.dump(story_embeddings, handle)

        return story_embeddings


class BertEncoder(TextEncoder):
    def __init__(self, modeldir, layer_indexes):
        self.vocab_file = modeldir + "/vocab.txt"
        self.config_file = modeldir + "/bert_config.json"
        self.init_checkpoint = modeldir + "/bert_model.ckpt"

        self.layer_indexes = layer_indexes

        self.bert_config = bert.modeling.BertConfig.from_json_file(self.config_file)

        self.tokenizer = bert.tokenization.FullTokenizer(
            vocab_file=self.vocab_file, do_lower_case=False)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        self.run_config = tf.contrib.tpu.RunConfig(
            master=None,
            tpu_config=tf.contrib.tpu.TPUConfig(
                num_shards=None,
                per_host_input_for_training=is_per_host))

    # What is happening here?
    def convert_example(self, input_sequences):
        examples = []
        unique_id = 0
        for sequence in input_sequences:
            # Tokenize
            line = bert.tokenization.convert_to_unicode(sequence)
            line = line.strip()

            # Two sequences, is that necessary for our task? I thought this was just for question-answering
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                extract_features.InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1

        return examples

    def get_embeddings_values(self, text_sequences, sequences_length, key=None):
        examples = self.convert_example(text_sequences)

        features = extract_features.convert_examples_to_features(
            examples=examples, seq_length=256, tokenizer=self.tokenizer)

        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature

        model_fn = extract_features.model_fn_builder(
            bert_config=self.bert_config,
            init_checkpoint=self.init_checkpoint,
            layer_indexes=self.layer_indexes,
            use_tpu=False,
            use_one_hot_embeddings=False)

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=self.run_config,
            predict_batch_size=32)

        input_fn = extract_features.input_fn_builder(
            features=features, seq_length=256)

        output_embeddings = []
        for result in estimator.predict(input_fn, yield_single_examples=True):
            unique_id = int(result["unique_id"])
            feature = unique_id_to_feature[unique_id]

            all_features = []
            for (i, token) in enumerate(feature.tokens):
                all_layers = []
                for (j, layer_index) in enumerate(self.layer_indexes):
                    layer_output = result["layer_output_%d" % j]
                    layers = collections.OrderedDict()
                    layers["index"] = layer_index
                    layers["values"] = [
                        round(float(x), 6) for x in layer_output[i:(i + 1)].flat
                    ]

                    all_layers.append(layers)
                features = collections.OrderedDict()
                features["token"] = token
                features["layers"] = all_layers
                all_features.append(features)
            output_embeddings.append(all_features)
        return output_embeddings