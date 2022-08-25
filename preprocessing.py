from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Preprocessing(object):
    def basic_preprocess(self, text):
        stops = stopwords.words("english")
        text = str(text).lower()
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in stops and len(token) > 2]
        text = " ".join(tokens)
        return text
