import pandas as pd
import numpy as np
from feature_generation import FeatureGeneration
from generate_arff import pandas2arff


morbidities = ['Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes', 'Gallstones', 'GERD', 'Gout', 'Hypercholesterolemia', 'Hypertension', 'Hypertriglyceridemia', 'OA', 'Obesity', 'OSA', 'PVD', 'Venous_Insufficiency']

for morbidity in morbidities:
    print(morbidity)
    train_preprocessed_df = pd.read_csv('./dataset/train/train_intuitive_preprocessed.csv')[['id', 'text', morbidity]]
    train_preprocessed_df = train_preprocessed_df[train_preprocessed_df[morbidity].isin([1.0, 0.0])]

    test_preprocessed_df = pd.read_csv('./dataset/test/test_intuitive_preprocessed.csv')[['id', 'text', morbidity]]
    test_preprocessed_df = test_preprocessed_df[test_preprocessed_df[morbidity].isin([1.0, 0.0])]

    
    X_train, Y_train, words_train = FeatureGeneration(train_preprocessed_df, morbidity).word2vec()
    
    
    X_train = np.average(X_train, axis=1)

    X = np.column_stack((X_train, Y_train))
    no_of_columns = X_train.shape[1]
    columns = ['f' + str(i) for i in range(no_of_columns)] + ['class']
    pandas2arff(pd.DataFrame(X, columns=columns), f'./dataset/train/train_{morbidity}_we.arff')