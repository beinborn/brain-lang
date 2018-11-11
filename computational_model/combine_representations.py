import numpy as np

# Variant a: combine word embeddings into a sentence embedding
# Variant b: combine sentence embeddings into a story embedding (for Kaplan data)
# TODO: a and b probably require different methods

#def combine_wordToSentence(stimulus, embedding):
 #   # TODO Samira: add your weighted averaging methods here

def combine_sentenceToStory(stimulus, embedding):
    if not len(stimulus) == len(embedding):
        raise ValueError("Stimulus and embedding should both be lists of same length")

    # Simple baseline, we should do something more sophisticated here like positional averaging
    concatenated_embedding = []
    for e in embedding:
        np.concatenate(concatenated_embedding, e)
    return concatenated_embedding

# So far, this method only works for the Harry Potter data and for elmo embeddings.
# It is absolutely not nice.
# TODO needs to be refactored!
def get_stimulus_embeddings(sentences, sentence_embeddings, scans):
    stimulus_embeddings = []

    for event in scans:
        sentence_id = len(event.sentences)
        token_id = len(event.current_sentence) - len(event.stimulus)
        token_embeddings = []

        # Right now, everything is aligned correctly, but that involved a lot of fiddling and should be done in a better way.
        if (len(event.stimulus) > 0):

            for token in event.stimulus:
                # We need to make some adjustments if there is a sentence boundary within the stimulus.
                while token_id < 0:
                    # Get last tokens from previous sentence
                    sentence_id = sentence_id - 1
                    token_id = len(sentences[sentence_id]) + token_id
                if token_id >= len(sentences[sentence_id]):
                    sentence_id += 1
                    token_id = 0

                # Make sure everything is aligned correctly
                print("Sentence id: "+ str(sentence_id))
                print("Token id: " + str(token_id))
                print("Stimulus token: " + token)
                if not (token.strip() == sentences[sentence_id][token_id].strip()):
                    print("Mismatch: ")
                    print(token)
                    print(sentences[sentence_id][token_id].strip())

                # I am taking the embedding from layer 1 here, this could be changed
                sentence_embedding = sentence_embeddings[sentence_id]
                token_embeddings.append(sentence_embedding[1][token_id])
                token_id += 1

            # TODO: here we would have different methods how to combine the token embeddings into a stimulus embedding
            # THis should be a parameter. Examples!
            # Option 1: Concatenate (need to find a solution to have uniform length: stimuli have different number of tokens
            # Option 2: Average
            # Option 3: Only take forward lm layer of last token (chosen here)
            stimulus_embeddings.append(token_embeddings[-1])
        # stimulus_embeddings.append(np.average(np.matrix(token_embeddings)))
        else:
            stimulus_embeddings.append([])
    return stimulus_embeddings

    # Very naive approach to the delay, just move the embeddings two timesteps down and skip the last two scans
    # TODO Discuss implementation of better methods with Samira!
def add_delay(timesteps, scans, embeddings):
    seed = 42
    empty_embedding = np.random.rand(embeddings[0].shape)

    embeddings = [[empty_embedding] * timesteps] + embeddings
    scans = scans[0:(len(scans) - 2)]
    if not len(embeddings) == len(scans):
        print("Embeddings and scans do not have the same length")
        print(len(embeddings))
        print(len(scans))
    return embeddings, scans
