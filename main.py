from pre_processing import *
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from gensim.models import Word2Vec
MAX_FEATURES=600

all_diseases = ['Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes', 'Gallstones',
            'GERD', 'Gout', 'Hypercholesterolemia', 'Hypertension', 
           'Hypertriglyceridemia', 'OA', 'Obesity', 'OSA', 'PVD', 'Venous Insufficiency']

def tf_idf(data, name):
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)

    docs = data['text'].values
    tfidf_matrix = vectorizer.fit_transform(docs)

    words = vectorizer.get_feature_names_out()

    X = tfidf_matrix.toarray()
    Y = np.array(data[name].values)
    print(X.shape, Y.shape, collections.Counter(list(Y)))
    return X, Y, words

def word2vec(data, name):
    sentences = data['text'].apply(lambda x: x.split(' ')).values

    model = Word2Vec(sentences, vector_size=10, window=5, min_count=1, workers=4)

    X = []

    for sentence in sentences:
        word_vectors = []
        for word in sentence:
            word_vectors.append(model.wv.get_vector(word))
        X.append(word_vectors)
    X = np.array(X)
    Y = np.array(data[name].values)

    words = model.wv.key_to_index.keys()
    
    return X, Y, words


def main():
    preprocess = PreProcess('intuitive', 'Asthma')

    tf_idf_matrix = tf_idf(preprocess.df, 'Asthma')
    word2vec_matrix = word2vec(preprocess.df, 'Asthma')



if __name__ == '__main__':
    main()