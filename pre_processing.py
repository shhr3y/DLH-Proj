import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

class PreProcess:
    def __init__(self, input, output) -> None:
        self.df = pd.read_csv(f'{input}.csv')
        # Initialize the lemmatizer
        self.lemmatizer = WordNetLemmatizer()


        self.df = self.df.apply(self.lowercasing, axis=1)
        self.df = self.df.apply(self.tokenize, axis=1)
        self.df = self.df.apply(self.remove_alph_num, axis=1)

        self.df = self.df.apply(self.lemmatize_sentence, axis=1)
        # print(self.df)
        self.df = self.df.apply(self.one_hot_encoding, axis=1)
        self.df.to_csv(f'{output}.csv', index=False)


    def lowercasing(self, row):
        row[1] = str.lower(row[1])
        return row

    def tokenize(self, row):
        row[1] = nltk.word_tokenize(row[1])
        return row

    def remove_alph_num(self, row):
        row[1] = list(filter(lambda x: x.isalpha(), row[1]))
        return row
    
    def one_hot_encoding(self, row):
        for row_idx in range(2, len(row)):
            # print(row[row_idx])
            if row[row_idx] == 'Y':
                row[row_idx] = 1.0
            elif row[row_idx] == 'N':
                row[row_idx] = 0.0
            else:
                row[row_idx] = -1
        return row
        
    def penn2morphy(self, penntag):
        """ Converts Penn Treebank tags to WordNet. """
        morphy_tag = {'NN':'n', 'JJ':'a',
                    'VB':'v', 'RB':'r'}
        try:
            return morphy_tag[penntag[:2]]
        except:
            return 'n' 

    def lemmatize_sentence(self, row):
        # Lemmatize each word, handling variations of "ing" and "ed" suffixes
        lemmatized_words = [self.lemmatizer.lemmatize(word, pos=self.penn2morphy(tag)) for word, tag in pos_tag(row[1])]
        # Join the lemmatized words back into a sentence
        row[1] = " ".join(lemmatized_words)
        # print(" ".join(lemmatized_words))
        return row


