from sentence_encoder import TextEmbedder
from faiss_index import IndexBuilder
import pandas as pd
import numpy as np
from time import time
import csv

class SearchEngine(object):
    def __init__(self, use_pretrained=False):
        self.embedder_model = TextEmbedder()
        self.faiss_index_obj = IndexBuilder()
        self.use_pretrained = use_pretrained
        self.search_df = pd.read_csv(
            "MovieSummaries/movie_dataset.csv"
        )
        if self.use_pretrained:
            self.faiss_index_pretrained = self.faiss_index_obj.read_index()

    def fit(self, X):
        embeddings = self.embedder_model.fit_transform(X=X)
        self.faiss_index_fit = IndexBuilder().build_index(
            X_embeddings=embeddings, dimensions=768, index_length=len(X)
        )

    def search(self, query):
        start = time()
        query_vector = self.embedder_model.fit_transform(query)
        if not self.use_pretrained:
            top_k = self.faiss_index_fit.search(query_vector, 5)
        else:
            top_k = self.faiss_index_pretrained.search(query_vector, 5)

        top_k_ids = top_k[1].tolist()[0]
        top_k_ids = list(np.unique(top_k_ids))
        results = [self.search_df["title"].iloc[idx] for idx in top_k_ids]
        end = time()
        print(f"{end-start} seconds for search")
        with open('query_results.txt','w') as file:
            for result in results:
                file.write(result)
                file.write('\n')
        return results


if __name__ == "__main__":
    import faiss

    index = faiss.read_index("faiss_indices/search_index.index")
    df = pd.read_csv("MovieSummaries/movie_dataset.csv")

    SE = SearchEngine()
    SE.fit(X=list(df["composite_doc"]))
    SE.search(query='Peter Parker')

