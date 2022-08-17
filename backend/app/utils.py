import re
import dill
from pathlib import Path
from abc import ABC

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

STOPWORDS = set(stopwords.words('russian'))
MIN_WORDS = 4
MAX_WORDS = 200

PATTERN_S = re.compile("\'s")  # matches `'s` from text  
PATTERN_RN = re.compile("\\r\\n") #matches `\r` and `\n`
PATTERN_PUNC = re.compile(r"[^\w\s]") # matches all non 0-9 A-z whitespace 

idfs = dill.load(open(Path('./data/vec_idf.dill'), "rb"))


class TfidfVectorizerCorr(TfidfVectorizer):
    TfidfVectorizer.idf_ = idfs


class ModelInference(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.tfidf_mat = self.load_resources(path='./data/tfidf.dill')
        self.token_stop = self.tokenizer(' '.join(STOPWORDS), lemmatize=False)
        self.vectorizer = self.create_vectorizer()
        self.items = pd.read_feather(Path('./data/item_data.feather'))

    def create_vectorizer(self):
        idf_diag = self.load_resources(path='./data/vec_idf_diag.dill')
        vocabulary = self.load_resources(path='./data/vec_voc.dill')
        vectorizer = TfidfVectorizerCorr(stop_words=self.token_stop, tokenizer=self.tokenizer, ngram_range=(1,2)) 
        vectorizer.vocabulary_ = vocabulary
        vectorizer._tfidf._idf_diag = idf_diag # sp.spdiags(idfs, diags=0, m=len(idfs), n=len(idfs))
        return vectorizer
        
    @staticmethod
    def load_resources(path):
        return dill.load(open(Path(path), "rb"))

    @staticmethod
    def tokenizer(sentence, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=STOPWORDS, lemmatize=True):
        """
        Lemmatize, tokenize, crop and remove stop words.
        """
        if lemmatize:
            stemmer = WordNetLemmatizer()
            tokens = [stemmer.lemmatize(w) for w in word_tokenize(sentence, language='russian')]
        else:
            tokens = [w for w in word_tokenize(sentence)]
        token = [w for w in tokens if (len(w) > min_words and len(w) < max_words
                                                            and w not in stopwords)]
        return tokens   


    @staticmethod
    def extract_best_indices(m, topk, mask=None):
        """
        Use sum of the cosine distance over all tokens.
        m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
        topk (int): number of indices to return (from high to lowest in order)
        """
        # return the sum on all tokens of cosinus for each sentence
        if len(m.shape) > 1:
            cos_sim = np.mean(m, axis=0) 
        else: 
            cos_sim = m
        index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score 
        if mask is not None:
            assert mask.shape == m.shape
            mask = mask[index]
        else:
            mask = np.ones(len(cos_sim))
        mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
        best_index = index[mask][:topk]  
        return best_index

    def create_json(self, best_index):
        cols = ['restaurant_name', 'street_adress', 'avg_rating']
        result = self.items[cols].iloc[best_index].to_dict('records')
        return json.dumps(result, ensure_ascii=False).encode('utf8')

    def get_prediction(self, sentence):
            
        """
        Return the database sentences in order of highest cosine similarity relatively to each 
        token of the target sentence. 
        """
        # Embed the query sentence
        tokens = [str(tok) for tok in self.tokenizer(sentence)]
        vec = self.vectorizer.transform(tokens)
        # Create list with similarity between query and dataset
        mat = cosine_similarity(vec, self.tfidf_mat)
        # Best cosine distance for each token independantly
        best_index = self.extract_best_indices(mat, topk=5)
        return self.create_json(best_index)
