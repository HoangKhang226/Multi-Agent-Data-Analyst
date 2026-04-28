import numpy as np

class FakeEmbedding:
    def __init__(self, dim=8):
        self.embed_dim = dim

    def get_text_embedding(self, text: str):
        np.random.seed(len(text))
        return np.random.rand(self.embed_dim).tolist()