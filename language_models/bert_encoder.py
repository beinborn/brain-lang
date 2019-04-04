
import pickle
import os
import logging
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from language_models_old.text_encoder import TextEncoder

# This class retrieves embeddings from the multilingual BertEncoder
# This was not part of the Cicling submission and has not yet been thoroughly tested.
class BertEncoder(TextEncoder):
    def __init__(self, embedding_dir, model_name="bert-base-multilingual-cased"):
        super(BertEncoder, self).__init__(embedding_dir)

        # Load pre-trained model (weights) and set to evaluation mode (no more training)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

        # Load word piece tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    # It does not make sense to get word embeddings without context from Bert
    # because we do not know which language they would come from.

    def get_sentence_embeddings(self, name, sentences, layer=-2):
        sentence_embeddings = []


        print("Number of sentences: " + str(len(sentences)))
        for sentence in sentences:
            # Bert uses its own "word piece tokenization"
            # It does not make sense to tokenize in the reader, then detokenize here and then tokenize again.
            # If I go on with that, I should probably not do tokenization in the reader.
            untokenized = " ".join(sentence)
            tokenized = self.tokenizer.tokenize(untokenized)
            # Convert token to vocabulary indices
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized)


            # double-check if this is the way to go for a single sentence
            segment_ids = [0 for token in tokenized]


            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensor = torch.tensor([segment_ids])

            # Predict hidden states features for each layer
            encoded_layers, _ = self.model(tokens_tensor, segments_tensor)

            assert len(encoded_layers) == 12

            # Which layer and which pooling function should we use for fixed sentence reperesentations?
            # 1. Jacob Devlin on: https://github.com/google-research/bert/issues/71
            # "If you want sentence representation that you don't want to train,
            # your best bet would just to be to average all the final hidden layers of all of the tokens in the sentence
            #  (or second-to-last hidden layers, i.e., -2, would be better)."
            # 2. In the paper, they say that concatenating the top four layers for each token could also be a good representation.
            # 3. In Bert as a service, they use the second to last layer and do mean pooling
            sentence_embeddings.append(encoded_layers[-1])
        return sentence_embeddings

    def get_story_embeddings(self, name, stories, mode="mean"):
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
                pooled_sentence_embeddings = []
                # Get a sentence embedding by pooling over all token embeddings
                # TODO: not yet sure what is the best choice here for layer and pooling method
                for embedding in sentence_embeddings:
                    if mode == "mean":
                        pooled_sentence_embeddings.append(embedding[0].mean(dim=0))
                    # Don't use this for Bert, not recommended
                    if mode == "sentence_final":
                        pooled_sentence_embeddings.append(embedding[-1])

                # Take the mean over all sentence embeddings
                # TODO: same here, unclear which is the best combination method
                # TODO: instead of concatenation of list, directly keep as tensor
                # combined should be of shape (number of sentences, 768))
                combined = torch.stack(pooled_sentence_embeddings,0)
                story_embedding = combined.mean(dim = 0)
                story_embeddings.append(story_embedding.detach().numpy())
                i += 1

            # Save embeddings
            os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
            with open(embedding_file, 'wb') as handle:
                pickle.dump(story_embeddings, handle)

        return story_embeddings



