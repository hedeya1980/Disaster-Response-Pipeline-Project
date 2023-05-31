import sys
from sklearn.base import BaseEstimator, TransformerMixin
import sqlite3
import pandas as pd
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download(['punkt', 'wordnet'])
nltk.download('averaged_perceptron_tagger')


def load_data(database_filepath):
    # connect to the database
    conn = sqlite3.connect(database_filepath)

    # run a query
    df=pd.read_sql('SELECT * FROM MessageCategory', conn)
    X = df[['message', 'genre']]#.values
    Y = df.iloc[:,4:]#.values
    
    target_cols=list(df.columns[4:])
    
    return X, Y, target_cols
    

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    
class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        if self.key=='genre':
            return data_dict[[self.key]]
        else:
            return data_dict[self.key]

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP']:
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

class MessageLengthExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_len = pd.Series(X).apply(lambda x: len(x))
        return pd.DataFrame(X_len)
        

def build_model():
    message_transformer = Pipeline([
        ('selector', ItemSelector(key='message')),
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range= (1, 2))),
        ('tdfidf', TfidfTransformer())])

    genre_transformer = Pipeline([
        ('selector', ItemSelector(key='genre')),
        ('onehot', OneHotEncoder())])

    starting_vb_extractor = Pipeline([
        ('selector', ItemSelector(key='message')),
        ('vb_extractor', StartingVerbExtractor())])

    msg_len_extractor = Pipeline([
        ('selector', ItemSelector(key='message')),
        ('msg_len_extractor', MessageLengthExtractor())])

    model=Pipeline([
        ('features', FeatureUnion([
            ('message_pipe', message_transformer),
            ('genre_pipe', genre_transformer),
            ('vb_extrctr', starting_vb_extractor),
            ('msg_len_extrctr', msg_len_extractor)
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier(min_samples_split=2, n_estimators=200, verbose=1)))
        #('clf', MultiOutputClassifier(LogisticRegression()))
    ])
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    # predict on test data
    y_pred = model.predict(X_test)

    report = classification_report(Y_test, y_pred, target_names=category_names)
    print(report)

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        # Some target columns has one unique value (either 1 or 0). For example, 'child_alone' has one unique value '0'. Need to amend any entry to be '1', to avoid errors with some classification models (Logistic Regression requires at least two classes in the label; otherwise, it won't work).
        single_value_targets=[]
        for col in category_names:#df.iloc[:,4:].columns:
            if len(Y_train[col].unique())==1:
                #print(col, df[col].unique(), len(df[col].unique()), category_names.index(col))
                single_value_targets.append(category_names.index(col))
                
        # impute the first entry of 'child_alone' to be 1 instead of 0
        for i in single_value_targets:
            Y_train.iloc[0,i]=1
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()