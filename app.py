from flask import *
import os

app = Flask(__name__)


# Library
# Library
import random
import time
import re
import nltk
import seaborn as sns
import numpy as np
import pandas as pd
import warnings

import os

from sklearn import preprocessing
from textblob import TextBlob
from deep_translator import GoogleTranslator
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from textblob import TextBlob

nltk.download('stopwords')
pd.set_option("display.max_columns", None)
preprocessing.LabelEncoder()
warnings.filterwarnings("ignore")
# 

SAVED_FOLDER = './saved-files'
app.config['SAVED_FOLDER'] = SAVED_FOLDER

df_ori = pd.read_csv(os.path.join(app.config['SAVED_FOLDER'], 'scrape_tweets_efek_vaksin_satria.csv'))
df_ori_head = df_ori.head()
replace_word = pd.read_csv(os.path.join(app.config['SAVED_FOLDER'], 'replace_word_list.csv'))
df_textCleansing = pd.read_csv(os.path.join(app.config['SAVED_FOLDER'], 'tweetCleansing.csv'))
df_textTranslated = pd.read_csv(os.path.join(app.config['SAVED_FOLDER'], 'tweetsTranslated.csv'))
df_textScoring = pd.read_csv(os.path.join(app.config['SAVED_FOLDER'], 'tweetScore.csv'))

df_textTranslate = df_textTranslated[['normalize_text', 'Translated']]



def preprocessing(dataset):
    sentiment = dataset[['Text']]
    df = sentiment.copy()
    df['cleansing'] = df['Text'].str.replace('@[^\s]+','')
    df['cleansing'] = df['cleansing'].str.replace('(#[A-Za-z0-9]+)','')
    df['cleansing'] = df['cleansing'].str.replace('http\s+','')
    df['cleansing'] = df['cleansing'].str.replace('(\w*\d\w*)','')
    df['cleansing'] = df['cleansing'].str.replace('&amp;',' ')
    df['cleansing'] = df['cleansing'].str.replace('[^A-Za-z\s\/]',' ')
    df['cleansing'] = df['cleansing'].str.replace('[^\w\s]',' ')
    df['cleansing'] = df['cleansing'].str.replace('\s+',' ')

    df['case_folding'] = df['cleansing'].apply(lambda x: x.lower())

    replace_word_dict = {}
    for i in range(replace_word.shape[0]):
        replace_word_dict[replace_word['before'][i]] = replace_word['after'][i]

    df['normalize_text'] = df['case_folding'].apply(lambda x : ' '.join(replace_word_dict.get(i, i) for i in x.split()), 1)
    df['tokenization'] = df['normalize_text'].apply(lambda x: x.split())

    factory = StopWordRemoverFactory()
    sastrawi_stopwords = factory.get_stop_words()
    df['stopword_removal'] = df['tokenization'].apply(lambda x: [word for word in x if word not in sastrawi_stopwords])

    df_cleansing = df[['Text', 'cleansing']]
    df_case_folding = df[['cleansing', 'case_folding']]
    df_normalize_text = df[['case_folding', 'normalize_text']]
    df_tokenization = df[['normalize_text', 'tokenization']]
    df_stopword_removal = df[['tokenization', 'stopword_removal']]

    return df_cleansing, df_case_folding, df_normalize_text, df_tokenization, df_stopword_removal, df

def scoring(dataset):
    data = dataset['Translated']
    
    def polarity(data):
        blob = TextBlob(data)
        return blob.polarity
    
    def getAnalysis(score):
        if score < 0 :
            return 'Negative'
        else:
            return 'Positive'
    
    dataset['polarity_score'] = dataset['Translated'].apply(polarity)
    dataset['analysis_result'] = dataset['polarity_score'].apply(getAnalysis)
    data = dataset[['Translated', 'polarity_score', 'analysis_result']]
    
    return data

def modelling(dataset):
    df = dataset
    X = df.drop(['analysis_result'], axis=1)
    y = df['analysis_result']
    
    from sklearn.preprocessing import LabelEncoder
    X = X.values
    le = LabelEncoder()
    le.fit(['Positive', 'Negative'])
    
    y = le.transform(y.values)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidfvectorizer = TfidfVectorizer(max_features=2000, min_df=5, stop_words=stopwords.words('english'), max_df=0.7, ngram_range=(1,3))
    X1 = tfidfvectorizer.fit_transform(df['Translated']).toarray()
    
    oversample =SMOTE()
    X1, y = oversample.fit_resample(X1, y)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=0)
    
    from sklearn.ensemble import RandomForestClassifier
    text_classifier_rf = RandomForestClassifier(n_estimators=100, random_state=0)
    t0_rf = time.time()
    text_classifier_rf.fit(X_train, y_train)
    t1_rf = time.time()
    
    predictions_rf = text_classifier_rf.predict(X_test)
    t2_rf = time.time()
    time_train_rf = t1_rf - t0_rf
    time_predict_rf = t2_rf - t1_rf
    
    from sklearn.naive_bayes import GaussianNB
    text_classifier_nb = GaussianNB()
    t0_nb = time.time()
    text_classifier_nb.fit(X_train, y_train)
    t1_nb = time.time()
    
    predictions_nb = text_classifier_nb.predict(X_test)
    t2_nb = time.time()
    time_train_nb = t1_nb - t0_nb
    time_predict_nb = t2_rf - t2_nb
    
    from sklearn.svm import SVC
    text_classifier_svm = SVC(kernel='linear')
    t0_svm = time.time()
    text_classifier_svm.fit(X_train, y_train)
    t1_svm = time.time()
    
    predictions_svm = text_classifier_svm.predict(X_test)
    t2_svm = time.time()
    time_train_svm = t1_svm - t0_svm
    time_predict_svm = t2_svm - t1_svm
    
    from sklearn.ensemble import StackingClassifier
    from lightgbm import LGBMClassifier
    
    modelLGB = LGBMClassifier(learning_rate=0.1)
    estimators = [
        ('rf', text_classifier_rf),
        ('nb', text_classifier_nb),
        ('svm', text_classifier_svm)
    ]
    text_classifier_stc = StackingClassifier(estimators=estimators, final_estimator=modelLGB)
    t0_stc = time.time()
    text_classifier_stc.fit(X_train, y_train)
    t1_stc = time.time()
    
    predictions_stc = text_classifier_stc.predict(X_test)
    t2_stc = time.time()
    time_train_stc = t1_stc - t0_stc
    time_predict_stc = t2_stc - t0_stc
    
    
    
    time_elapsed_svm = [time_train_svm, time_predict_svm]
    time_elapsed_nb = [time_train_nb, time_predict_nb]
    time_elapsed_rf = [time_train_rf, time_predict_rf]
    time_elapsed_stc = [time_train_stc, time_predict_stc]
    
    
    from sklearn.metrics import accuracy_score
    acc_rf = round(accuracy_score(y_test, predictions_rf)*100,2)
    acc_nb = round(accuracy_score(y_test, predictions_nb)*100,2)
    acc_svm = round(accuracy_score(y_test, predictions_svm)*100,2)
    acc_stc = round(accuracy_score(y_test, predictions_stc)*100,2)
    
    
    return acc_rf, acc_nb, acc_svm, acc_stc

modelling_result = modelling(df_textScoring)

# @app.route("/")
# def home():
#     data_html = df_ori_head.to_html(classes='table table-hover', justify='justify', index=False)
#     row = df_ori_head.columns
#     col = df_ori_head.values
#     return render_template("index.html", 
#                            table = data_html,
#                            row = row,
#                            col = col,
#                            )

@app.route('/')
def abstrak():
    return render_template("abstrak.html")

@app.route('/datasetOri')
def datasetOri():
    data_html = df_ori_head.to_html(classes='table table-hover', justify='justify', index=False)
    
    return render_template("datasetOri.html",
                           table = data_html
                           )

@app.route('/preprocessing')
def preprocessingResult():
    preprocessing_result = preprocessing(df_ori)
    df_cleansing = preprocessing_result[0].head().to_html(classes='table table-hover', justify='justify', index=False)   
    df_casefolding = preprocessing_result[1].head().to_html(classes='table table-hover', justify='justify', index=False)   
    df_normalize_text = preprocessing_result[2].head().to_html(classes='table table-hover', justify='justify', index=False)   
    df_tokenization = preprocessing_result[3].head().to_html(classes='table table-hover', justify='justify', index=False)   
    df_stopword_removal = preprocessing_result[4].head().to_html(classes='table table-hover', justify='justify', index=False)   
    df_translated = df_textTranslate.head().to_html(classes='table table-hover', justify='justify', index=False)
    
    scoring_result = scoring(df_textTranslated)
    df_scoring = scoring_result.head().to_html(classes='table table-hover', justify='justify', index=False)
    
    return render_template('preprocessing.html',
                           cleansing = df_cleansing,
                           casefolding = df_casefolding,
                           normalize_text = df_normalize_text,
                           tokenization = df_tokenization,
                           stopword_removal = df_stopword_removal,
                           translated = df_translated,
                           scoring = df_scoring)
    
@app.route('/result')
def modellingResult():
    rf_result = modelling_result[0]
    nb_result = modelling_result[1]
    svm_result = modelling_result[2]
    stc_result = modelling_result[3]
    all_result = modelling_result
    
    return render_template('result.html',
                           rf = rf_result,
                           nb = nb_result,
                           svm = svm_result,
                           stc = stc_result,
                           all = all_result)

# 
if __name__ == "__main__":
    app.run(debug=True)
