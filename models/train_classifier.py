import sys
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import pickle
from pprint import pprint
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine
import time
import warnings
warnings.filterwarnings('ignore')


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_messages', con=engine)
    X, Y = df['message'], df.iloc[:, 4:]

    # Y['related'] contains three distinct values
    # mapping extra values to `1`
    Y['related'] = Y['related'].map(lambda x: 1 if x == 2 else x)
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    word_lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = word_lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    # model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    pipeline.get_params()
    # parameters
    parameters = {'tfidf__use_idf': (True, False),
                  'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__min_samples_split': [2, 4]}
    # create model
    CV = GridSearchCV(pipeline, param_grid=parameters)
    return CV


def evaluate_model(model, X_test, Y_test, category_names):
    # predict
    predict = model.predict(X_test)
    # print classification report
    print(classification_report(Y_test.values,
          predict, target_names=category_names))
    # print accuracy score
    print('Accuracy: {}'.format(np.mean(Y_test.values == predict)))


def save_model(model, model_filepath):
    # save model to pickle file
    pickle.dump(model, open(model_filepath, 'wb')

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath=sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names=load_data(database_filepath)
        X_train, X_test, Y_train, Y_test=train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model=build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
