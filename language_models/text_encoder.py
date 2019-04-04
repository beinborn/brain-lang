#TODO: adjust this if necessary


class TextEncoder(object):
    def __init__(self, embedding_dir):
        self.embedding_dir = embedding_dir

    def get_embeddings(self, text, sequences_length):
        raise NotImplementedError()

