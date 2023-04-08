import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

class PreProcess:
    def __init__(self, filename, disease) -> None:
        self.df = pd.read_csv(f'{filename}.csv', usecols= ['id', 'text', disease])
        self.df = self.df[self.df[disease].isin(['Y', 'N'])]
        # Initialize the lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        # self.lowercasing()
        self.df['text'].apply(self.lemmatize_sentence)

    

    def lowercasing(self):
        self.df['text'] = self.df['text'].str.lower()

    def tokenize(self):
        self.df['text'] = self.df['text'].str.split(' ')

    def alph_num(self):
        values = list("abcdefghijklmnopqrstuvwxyz ")
        text = self.df['text']

        for c in text:
            if c not in values:
                text = text.replace(c, "")
        
        self.df['text'] = text
        
    def penn2morphy(self, penntag):
        """ Converts Penn Treebank tags to WordNet. """
        morphy_tag = {'NN':'n', 'JJ':'a',
                    'VB':'v', 'RB':'r'}
        try:
            return morphy_tag[penntag[:2]]
        except:
            return 'n' 
    def lemmatize_sentence(self, sentence):
        # Tokenize the sentence into individual words
        # item = self.df['text'].
        # print(item)

        words = nltk.word_tokenize(sentence)
        # Lemmatize each word, handling variations of "ing" and "ed" suffixes
        lemmatized_words = [self.lemmatizer.lemmatize(word.lower(), pos=self.penn2morphy(tag)) for word, tag in pos_tag(words)]
        # Join the lemmatized words back into a sentence
        lemmatized_sentence = " ".join(lemmatized_words)
        print(lemmatized_sentence)



        


