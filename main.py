from pre_processing import *
from feature_generation import *



def main():
    preprocess = PreProcess('intuitive', 'Asthma')

    tf_idf_matrix = tf_idf(preprocess.df, 'Asthma')
    word2vec_matrix = word2vec(preprocess.df, 'Asthma')



if __name__ == '__main__':
    main()