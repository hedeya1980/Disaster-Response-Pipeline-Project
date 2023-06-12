#from responseapp import app

import json
import plotly
import pandas as pd
import sqlite3

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
#from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin


#from app.classDefs import ItemSelector, StartingVerbExtractor, MessageLengthExtractor

app = Flask(__name__)

def tokenize(text):
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
'''
# load data
#engine = create_engine('sqlite:///../data/DisasterResponse.db')
#df = pd.read_sql_table('MessageCategory', engine)

# connect to the database
conn = sqlite3.connect('./data/DisasterResponse.db')

# run a query
df=pd.read_sql('SELECT * FROM MessageCategory', conn)

# load model
#model = joblib.load("../models/msg_gnre_pipeline.pkl")
#model = joblib.load("../models/more_features_model.pkl")
model = joblib.load("./models/cv_compressed_model.pkl")
'''
#def main():
#app.run(host='0.0.0.0', port=3000, debug=True)
# load data
#engine = create_engine('sqlite:///../data/DisasterResponse.db')
#df = pd.read_sql_table('MessageCategory', engine)

# connect to the database
conn = sqlite3.connect('./data/DisasterResponse.db')

# run a query
df=pd.read_sql('SELECT * FROM MessageCategory', conn)

# load model
#model = joblib.load("../models/msg_gnre_pipeline.pkl")
#model = joblib.load("../models/more_features_model.pkl")
model = joblib.load("./models/cv_compressed_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_counts=df.iloc[:,4:].sum(axis=0)
    category_names=list(df.columns)[4:]
    
    category_by_genre=df.groupby('genre').sum().iloc[:,4:].transpose().sort_values('direct', ascending=False)
    direct_cat_counts=df[df['genre']=='direct'].iloc[:,4:].sum(axis=0)
    news_cat_counts=df[df['genre']=='news'].iloc[:,4:].sum(axis=0)
    social_cat_counts=df[df['genre']=='social'].iloc[:,4:].sum(axis=0)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
                {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
                {
            'data': [
                Bar(
                    name='direct',
                    x=category_names,
                    y=direct_cat_counts
                ),
                Bar(
                    name='news',
                    x=category_names,
                    y=news_cat_counts
                ),
                Bar(
                    name='social',
                    x=category_names,
                    y=social_cat_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories by Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    genre = request.args.get('genre', 'direct')    

    # use model to predict classification for query
    #print(query)
    #classification_labels = model.predict([query])[0]
    classification_labels = model.predict(pd.DataFrame(data={'message': query, 'genre':genre}, columns=['message', 'genre'], index=[1]))[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )



if __name__ == '__main__':
    main()