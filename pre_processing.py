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

        self.lowercasing()
        self.df['text'] = self.df['text'].apply(self.alph_num)
        self.df['text'] = self.df['text'].apply(self.lemmatize_sentence)


    def lowercasing(self):
        self.df['text'] = self.df['text'].str.lower()

    def tokenize(self):
        self.df['text'] = self.df['text'].str.split(' ')

    def alph_num(self, sentence):
        values = list("abcdefghijklmnopqrstuvwxyz ")

        for c in sentence:
            if c not in values:
                sentence = sentence.replace(c, "")
        
        return sentence
        
    def penn2morphy(self, penntag):
        """ Converts Penn Treebank tags to WordNet. """
        morphy_tag = {'NN':'n', 'JJ':'a',
                    'VB':'v', 'RB':'r'}
        try:
            return morphy_tag[penntag[:2]]
        except:
            return 'n' 

    def lemmatize_sentence(self, sentence):
        words = nltk.word_tokenize(sentence)
        # Lemmatize each word, handling variations of "ing" and "ed" suffixes
        lemmatized_words = [self.lemmatizer.lemmatize(word.lower(), pos=self.penn2morphy(tag)) for word, tag in pos_tag(words)]
        # Join the lemmatized words back into a sentence
        sentence = " ".join(lemmatized_words)
        return sentence



        


