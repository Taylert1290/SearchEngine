from sentence_transformers import SentenceTransformer
import numpy as np

from preprocessing import Preprocessing


class TextEmbedder(Preprocessing):
    """
    Encodes a list of text inputs
    """

    def __init__(self):
        self.model = SentenceTransformer("msmarco-distilbert-base-dot-prod-v3")

    def fit_transform(self, X: list) -> np.array:
        if not isinstance(X, list):
            X = [X]
        X = [self.basic_preprocess(text) for text in X]
        embeddings = self.model.encode(X, show_progress_bar=True)
        numpy_embedding = np.asarray(embeddings.astype("float32"))
        return numpy_embedding
