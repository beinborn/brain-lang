import numpy as np


# Variant a: combine word embeddings into a sentence embedding
# Variant b: combine sentence embeddings into a story embedding (for Kaplan data)
# TODO: a and b probably require different methods

# def combine_wordToSentence(stimulus, embedding):
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
                # print("Sentence id: "+ str(sentence_id))
                # print("Token id: " + str(token_id))
                # print("Stimulus token: " + token)
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

    # Very naive approach to the delay.
    # Just add random fixation embeddings to the beginning according to the number of timesteps n and skip the last n embeddings
    # TODO Discuss implementation of better methods with Samira!


def add_delay(timesteps, scans, embeddings):
    np.random.seed(42)
    embedding_dim = len(max(embeddings, key=len))
    fixation_embedding = np.random.rand(embedding_dim, )

    for i in range(0, len(embeddings)):
        if len(embeddings[i]) == 0:
            embeddings[i] = fixation_embedding

    # Add fixation embeddings to the beginning
    initial_embeddings = []
    for step in range(0, timesteps):
        initial_embeddings.append(fixation_embedding)

    embeddings = initial_embeddings + embeddings

    # Remove last two embeddings from end
    embeddings = embeddings[0:(len(embeddings) - timesteps)]

    if not len(embeddings) == len(scans):
        print("Embeddings and scans do not have the same length")
        print(len(embeddings))
        print(len(scans))
    else:
        print("All fine.")
    return scans, embeddings
