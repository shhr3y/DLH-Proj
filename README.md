# Ensembling Classical Machine Learning and Deep Learning Approaches for Morbidity Identification From Clinical Notes

This repository contains the code for the Research Paper: Ensembling Classical Machine Learning and Deep Learning Approaches for Morbidity Identification From Clinical Notes started for Final Project in CS 598 Deep Learning in Healthcare at University of Illinois Urbana-Champaign. This paper demonstrates and explores the use of ensembling techniques to combine classical machine learning and deep learning approaches for more accurate morbidity identification from clinical notes.


# Getting Started
In this paper we have used n2c2 NLP Dataset for which we had to request access from [Harvard Medical School Portal](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/). This dataset was originally generated during the i2b2 project.


### Prerequisites
Clone the repository
* `git clone https://github.com/shhr3y/DLH-Proj`

We highly recommend to use a virtual enviroment for this project as there might be some dependencies which might cause version conflicts. We are using [Miniforge](https://github.com/conda-forge/miniforge) for creating our environment.
* run `brew install miniforge` to install miniforge on your system.
* run `conda create --name dlh python=3.10`. (we recommend using python version 3.10 for this project)
* run `conda activate dlh` 

To install all the dependencies for this projects:
* run `pip install -r requirements.txt`

### Dataset Generation
Before proceeding, [Harvard Medical School Portal](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/) is in xml format which we need to convert to csv for training. To convert .xml data to .csv format, run `xml2csv.py` which will generate a csv file containing all columns such as [id, text, Asthma, CAD, CHF, ..]
* run `python xml2csv.py`

After generating the csv file, we need to apply our pre-processing functions (tokenization, lowercase, remove non-alpha chars, lemmatize, one-hot encode). For this purpose we can use PreProcess class with inputs as input_file_destination(generated csv file path) and output_file_destination(output csv file pash) which is in `pre_processing.py` which can be import into other scripts and used.

We have used Weka for some machine learning models such as J-Rip and J48. Weka tool needs its input as .arff file needs to be generated. We have used this [script for pandas2arff](https://github.com/saurabhnagrecha/Pandas-to-ARFF/blob/master/pandas2arff.py) and generated the required files needed by the Weka library. This function has been used to create dataset for TF-IDF and WordEmbedding representations. We have multiple word embedding techniques used (code in `feature_generation.py`) such as Word2Vec, GloVe, FastText, Universal Sentence Encoder.
* run `python screate_arff_tfidf.py` for generating TF-IDF dataset.
* run  `python create_arff_word_embeddings.py` for generating WordEmbeddings dataset.


These instructions will get you a all the dataset files required for the project ready on your local machine for development and testing purposes.

### Classical Machine Learning Models
We have implemented several Machine Learning Models such as DecisionTree, J-48, JRip, KNN, NaiveBayes, RandomForest and SVM. The code for the same models can be found under [`/ML/`](https://github.com/shhr3y/DLH-Proj/tree/main/ML) directory. Further, we have used two types of textual representations TF-IDF and WordEmbeddings which can be found at [`ML/tf-idf`](https://github.com/shhr3y/DLH-Proj/tree/main/ML/tf-idf) and [`ML/word-embeddings`](https://github.com/shhr3y/DLH-Proj/tree/main/ML/word-embeddings). 


### Deep Learning Model
We have implemented Stat Bidirectional Long-Short Term Memory (BiLSTM) Layer RNN with n = 2 and 128 and 64 hidden layers. We have used Binary Cross-Entrophy as our loss function and have used several word embeddings such as GloVe, FastText, UniversalSentenceEncodder, Word2Vec. The code for this deep-learning model can be found at [/DL/](https://github.com/shhr3y/DLH-Proj/tree/main/DL).


These instructions will get you a all the results of different models ready on your local machine for development and testing purposes.


# Results

## CML(RN)Preformance with SMOTE and ExtraTrees
We have acheived better results with RandomForest with SMOTE and ExtreTrees classifier for feature selection in TF-IDF when compared RandomForest with AllFeatures without any feature selection or over sampling in TF-IDF, which depicts that feature selection increases performace.


|Morbidity Class     |w/o SMOTE & ExtraTrees RFMacro F1     |w/o SMOTE & ExtraTrees  RF_Micro F1    |SMOTE & ExtraTrees RF_Macro F1       |SMOTE & ExtraTrees RF_Micro F1       |
|--------------------|--------------------|-------------------|------------------|------------------|
|Asthma              | 0.49245763267483306|0.8810949788263762 |0.9888628096539887|0.989019801980198 |
|CAD                 | 0.9178062504459759 |0.9233333333333335 |0.9266441020391909|0.9276923076923078|
|CHF                 | 1.0                |1.0                |1                 |1                 |
|Depression          | 0.5138524072645619 |0.8058153126826418 |0.9339450018380673|0.9347826086956521|
|Diabetes            | 0.8739219960299753 |0.901441102756892  |0.9771615914904839|0.9772943037974684|
|Gallstones          | 0.4599902140879871 |0.8532485875706215 |0.9294916354225806|0.9308289652494661|
|GERD                | 0.4323776782842145 |0.7638605442176871 |0.877896163015734 |0.879045045045045 |
|Gout                | 0.4647957966912661 |0.8691242937853108 |0.9548711662946365|0.955563853622106 |
|Hypercholesterolemia| 0.8400652419091632 |0.8425490196078431 |0.8845109536038042|0.8872641509433963|
|Hypertension        | 0.50500773716472   |0.81743535988819   |0.9610438529632302|0.9615184678522573|
|Hypertriglyceridemia| 0.4854267614497737 |0.9438924605493864 |0.9836073013237844|0.9837510237510237|
|OA                  | 0.46509042983465887|0.8282581453634086 |0.9257431596498952|0.9271219400594829|
|Obesity             | 0.9076256375579452 |0.913116883116883  |0.9771596670227283|0.9777009728622632|
|OSA                 | 0.5519158380668717 |0.8711864406779661 |0.9736740853175949|0.974296253154727 |
|PVD                 | 0.5783369875048958 |0.8632792207792208 |0.975896847083613 |0.9765614275909403|
|Venous_Insufficiency| 0.47801077753999144|0.9164368650217707 |0.9736143705713205|0.9741194158075602|
|Average             | 0.6229175866566771 |0.8746295342610958 |0.952757669205666 |0.9535350336314934|

## DL Performace with different word embeddings
After testing our deep learning models with different word embedding we found that USE(Universal Sentence Encoder).

|Embeddings|Avg Macro F1|Avg Micro F1|
|----------|------------|------------|
|Word2Vec  | 65.98      | 81.99      |
|Glove     | 63.87      | 78.95      |
|FastText  | 66.65      | 81.94      |
|USE       | 74.00      | 85.69      |


## Authors
**kshitij6798**  
**shhr3y**  
