import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class DataSetup(object):
    def get_data(self):
        ratings = pd.read_csv('../MovieSummaries/ratings.csv')
        movies = pd.read_csv('../MovieSummaries/movies.csv')
        ratings = pd.merge(movies,ratings,on='movieId')
        matrix =  ratings.pivot_table(index='userId',columns='title',values='rating').fillna(0)
        base_matrix = matrix.reset_index().rename_axis(None,axis=1)
        model_matrix = base_matrix.drop('userId',1)
        model_matrix.columns = matrix.columns
        return base_matrix, model_matrix
