from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import numpy as np
from gensim.models import Word2Vec
from gensim.models import FastText
import tensorflow_hub as hub
import collections
import os


MAX_FEATURES=600
VECTOR_SIZE = 300
DOCUMENT_LENGTH = 500


glove_file_path = './dataset/embeddings/glove.6B.300d.txt'

all_diseases = ['Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes', 'Gallstones',
            'GERD', 'Gout', 'Hypercholesterolemia', 'Hypertension', 
           'Hypertriglyceridemia', 'OA', 'Obesity', 'OSA', 'PVD', 'Venous Insufficiency']


class FeatureGeneration:
    def __init__(self, data, disease_name):
        self.data = data
        self.disease_name = disease_name

    def tf_idf(self):
        vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)

        docs = self.data['text'].values
        tfidf_matrix = vectorizer.fit_transform(docs)

        words = vectorizer.get_feature_names_out()

        X = tfidf_matrix.toarray()
        Y = np.array(self.data[self.disease_name].values)
        print(X.shape, Y.shape, collections.Counter(list(Y)))
        return X, Y, words

    def word2vec(self):
        # spliting text by ' '
        self.data['split_text'] = self.data['text'].apply(lambda x: x.split(' '))
        # filtering based on DOCUMENT_LENGTH 
        self.data = self.data[self.data.apply(lambda row: len(row['split_text']) < DOCUMENT_LENGTH, axis=1)]

        sentences = self.data['split_text'].values

        model = Word2Vec(sentences, vector_size=VECTOR_SIZE, window=5, min_count=1, workers=4)

        # finding max length after sorting
        max_length = sorted(sentences, key=lambda x: len(x), reverse=True)[0]

        X = np.zeros((len(sentences), max_length, VECTOR_SIZE)) 

        for idx, sentence in enumerate(sentences):
            sentence = [word if word in model.wv.key_to_index else 'UNK' for word in sentence]
            sentence_vectors = [model.wv[word] for word in sentence]
            sentence_vectors += [np.zeros(VECTOR_SIZE)] * (max_length - len(sentence))

            X[idx] = np.array(sentence_vectors)

        Y = np.array(self.data[self.disease_name].values)
        words = model.wv.key_to_index.keys()
        
        return X, Y, words
    
    def gloVe(self, max_length=100):
        word_vectors = {}
        with open(glove_file_path, encoding='utf8') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                word_vectors[word] = coefs
        word_vectors['UNK'] = np.random.rand(self.VECTOR_SIZE)

        # spliting text by ' '
        self.data['split_text'] = self.data['text'].apply(lambda x: x.split(' '))
        # filtering based on DOCUMENT_LENGTH 
        self.data = self.data[self.data.apply(lambda row: len(row['split_text']) < DOCUMENT_LENGTH, axis=1)]
        
        sentences = self.data['split_text'].values
        # taking top max_length words in sentence
        sentences = [s[:max_length] for s in sentences]

        X = np.zeros((len(sentences), max_length, self.VECTOR_SIZE))
        for idx, sentence in enumerate(sentences):
            sentence = [word if word in word_vectors else 'UNK' for word in sentence]
            sentence_vectors = [word_vectors.get(word, word_vectors['UNK']) for word in sentence]
            sentence_vectors += [np.zeros(self.VECTOR_SIZE)] * (max_length - len(sentence))

            X[idx, :, :] = np.array(sentence_vectors)
        
        Y = np.array(self.df[self.disease_name].values.tolist())
        words = list(word_vectors.keys())
        return X, Y, words
    
    def fastText(self):
        # spliting text by ' '
        self.data['split_text'] = self.data['text'].apply(lambda x: x.split(' '))
        # filtering based on DOCUMENT_LENGTH 
        self.data = self.data[self.data.apply(lambda row: len(row['split_text']) < DOCUMENT_LENGTH, axis=1)]

        sentences = self.data['split_text'].values

        fasttext_model = FastText(sentences, vector_size=VECTOR_SIZE, window=5, min_count=1, workers=4)

        max_length = max(len(sentence) for sentence in sentences)
        X = np.zeros((len(sentences), max_length, VECTOR_SIZE)) 

        for idx, sentence in enumerate(sentences):
            sentence = [word if word in fasttext_model.wv.key_to_index else 'UNK' for word in sentence]
            sentence_vectors = [fasttext_model.wv[word] for word in sentence]
            sentence_vectors += [np.zeros(VECTOR_SIZE)] * (max_length - len(sentence))
            
            X[idx] = np.array(sentence_vectors)

        Y = np.array(self.data[self.disease_name].values)
        words = fasttext_model.wv.key_to_index.keys()
        return X, Y, words

    def universal_sentence_encoder(self):
        # spliting text by ' '
        self.data['split_text'] = self.data['text'].apply(lambda x: x.split(' '))
        # filtering based on DOCUMENT_LENGTH 
        self.data = self.data[self.data.apply(lambda row: len(row['split_text']) < DOCUMENT_LENGTH, axis=1)]

        sentences = self.data['split_text'].apply(lambda x: ' '.join(x)).values 

        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        sentence_embeddings = embed(sentences)
        embedding_size = sentence_embeddings.shape[-1]


        projection_matrix = np.random.randn(embedding_size, VECTOR_SIZE)

        embeddings_300 = np.dot(sentence_embeddings, projection_matrix)

        num_sentences = len(sentences)
        Y = np.array(self.data[self.disease_name].values)
        X = np.reshape(embeddings_300, (num_sentences, 1, VECTOR_SIZE))

        return X, Y, []
