import pandas as pd
import numpy as np
import collections
from feature_generation import FeatureGeneration
from generate_arff import pandas2arff
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.filters import Filter
from weka.attribute_selection import ASEvaluation, AttributeSelection
from weka.classifiers import Classifier, Evaluation


jvm.start()

morbidities = ['Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes', 'Gallstones', 'GERD', 'Gout', 'Hypercholesterolemia', 'Hypertension', 'Hypertriglyceridemia', 'OA', 'Obesity', 'OSA', 'PVD', 'Venous_Insufficiency']

def feature_selection_SelectKBest(k, x_train, y_train):
    k_best = SelectKBest(chi2, k=k)
    k_best.fit(x_train, y_train)
    return k_best.transform(x_train)
    
def feature_selection_ExtraTreesClassifier(k, x_train, y_train):
    clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    return x_train[:, indices[:k]]

def feature_selection_InfoGainAttributeEval(x_train, y_train, morbidity):

    if len(collections.Counter(list(y_train)).keys()) < 2:
        return x_train

    loader = Loader(classname="weka.core.converters.ArffLoader")
    train_data = loader.load_file(f"./dataset/train/train_{morbidity}_tfidf.arff")
    train_data.class_is_last()

    eval = ASEvaluation(classname="weka.attributeSelection.InfoGainAttributeEval")
    search = AttributeSelection()
    search.evaluator = eval
    search.select_attributes(train_data)
    selected_attributes = search.selected_attributes
    filtered_attributes = np.delete(selected_attributes, [-1])

    return x_train[:, filtered_attributes]


for morbidity in morbidities:
    print(morbidity)
    train_preprocessed_df = pd.read_csv('./dataset/train/train_intuitive_preprocessed.csv')[['id', 'text', morbidity]]
    train_preprocessed_df = train_preprocessed_df[train_preprocessed_df[morbidity].isin([1.0, 0.0])]

    X_train, Y_train, words_train = FeatureGeneration(train_preprocessed_df, morbidity).tf_idf()
    
    X = np.column_stack((X_train, Y_train))
    no_of_columns = X_train.shape[1]
    columns = ['f' + str(i) for i in range(no_of_columns)] + ['class']
    pandas2arff(pd.DataFrame(X, columns=columns), f'./dataset/train/train_{morbidity}_tfidf.arff')


    X_train_SelectKBest = feature_selection_SelectKBest(100, X_train, Y_train)
    X = np.column_stack((X_train_SelectKBest, Y_train))
    no_of_columns = X_train_SelectKBest.shape[1]
    columns = ['f' + str(i) for i in range(no_of_columns)] + ['class']
    pandas2arff(pd.DataFrame(X, columns=columns), f'./dataset/train/train_{morbidity}_SelectKBest_tfidf.arff')


    X_train_ExtraTreesClassifier = feature_selection_ExtraTreesClassifier(100, X_train, Y_train)
    X = np.column_stack((X_train_ExtraTreesClassifier, Y_train))
    no_of_columns = X_train_ExtraTreesClassifier.shape[1]
    columns = ['f' + str(i) for i in range(no_of_columns)] + ['class']
    pandas2arff(pd.DataFrame(X, columns=columns), f'./dataset/train/train_{morbidity}_ExtraTreesClassifier_tfidf.arff')


    X_train_InfoGainAttributeEval = feature_selection_InfoGainAttributeEval(X_train, Y_train, morbidity)
    X = np.column_stack((X_train_InfoGainAttributeEval, Y_train))
    no_of_columns = X_train_InfoGainAttributeEval.shape[1]
    columns = ['f' + str(i) for i in range(no_of_columns)] + ['class']
    pandas2arff(pd.DataFrame(X, columns=columns), f'./dataset/train/train_{morbidity}_InfoGainAttributeEval_tfidf.arff')

jvm.stop()