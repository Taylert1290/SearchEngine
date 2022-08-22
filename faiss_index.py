import faiss
import numpy as np


class IndexBuilder(object):
    def build_index(self, X_embeddings, dimensions, index_length):
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dimensions))
        index.add_with_ids(X_embeddings, np.array(range(0, index_length)))
        faiss.write_index(index, "faiss_indices/search_index.index")
        return index

    def read_index(self):
        return faiss.read_index("faiss_indices/search_index.index")
