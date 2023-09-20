from gensim.models import Word2Vec
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('vader_lexicon')

file = pd.read_csv('dataset/op_spam.csv')
df = file
reviews = df['reviews']

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

preprocessed_reviews = []
sentiment_scores = []

for review in reviews:
    # Normalize text
    review = review.lower()
    # Remove numbers
    review = re.sub(r'\d+', '', review)
    # Tokenize text
    tokens = nltk.word_tokenize(review)
    # Remove punctuation and non-alphanumeric tokens
    tokens = [token for token in tokens if token.isalnum()]
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # Stemming
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join tokens back into text
    review_cleaned = ' '.join(tokens)
    # Sentiment score
    sentiment_scores.append(sia.polarity_scores(review_cleaned)['compound'])
    preprocessed_reviews.append(review_cleaned)

word2vec_model = Word2Vec([review.split() for review in preprocessed_reviews], vector_size=1000, window=5, min_count=5, workers=4)


def review_to_vector(tokens, model):
    vectors = [model.wv[token] for token in tokens if token in model.wv.index_to_key]
    if vectors:
        vector = sum(vectors) / len(vectors)
    else:
        vector = [0] * model.vector_size
    return vector


word2vec_vectors = [review_to_vector(review.split(), word2vec_model) for review in preprocessed_reviews]

scaler = MinMaxScaler()
scaled_word2vec_vectors = scaler.fit_transform(word2vec_vectors)

df = df.drop('reviews', axis=1)
df.insert(3, 'senti_score', sentiment_scores)
df.insert(3, 'reviews', preprocessed_reviews)
target = df['is_fake']
cols = [i for i in df.columns if i not in ['is_fake']]
data = df[cols]
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=5, max_features=1000)
tfidf_vectorizer.fit(X_train['reviews'])
train_vectors = tfidf_vectorizer.transform(X_train['reviews'])
test_vectors = tfidf_vectorizer.transform(X_test['reviews'])

train_df = pd.DataFrame(train_vectors.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
test_df = pd.DataFrame(test_vectors.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

train_punc = np.squeeze(scaler.fit_transform(np.array(X_train['punctuation_num']).reshape(-1, 1)))
train_cap = np.squeeze(scaler.fit_transform(np.array(X_train['capture_num']).reshape(-1, 1)))
train_senti = np.squeeze(scaler.fit_transform(np.array(X_train['senti_score']).reshape(-1, 1)))
train_length = np.squeeze(scaler.fit_transform(np.array(X_train['length']).reshape(-1, 1)))
train_df.insert(0, 'cap_num', train_cap)
train_df.insert(0, 'punc_num', train_punc)
train_df.insert(0, 'senti_score', train_senti)
train_df.insert(0, 'length', train_length)
train_df.insert(0, 'is_fake', y_train.reset_index(drop=True))
train_df.to_csv('train_tfidf.csv')

test_punc = np.squeeze(scaler.fit_transform(np.array(X_test['punctuation_num']).reshape(-1, 1)))
test_cap = np.squeeze(scaler.fit_transform(np.array(X_test['capture_num']).reshape(-1, 1)))
test_senti = np.squeeze(scaler.fit_transform(np.array(X_test['senti_score']).reshape(-1, 1)))
test_length = np.squeeze(scaler.fit_transform(np.array(X_test['length']).reshape(-1, 1)))
test_df.insert(0, 'cap_num', test_cap)
test_df.insert(0, 'punc_num', test_punc)
test_df.insert(0, 'senti_score', test_senti)
test_df.insert(0, 'length', test_length)
test_df.insert(0, 'is_fake', y_test.reset_index(drop=True))
test_df.to_csv('test_tfidf.csv')

train_word2vec_df = pd.DataFrame(scaled_word2vec_vectors[:len(X_train)], columns=[f'w2v_dim_{i}' for i in range(word2vec_model.vector_size)])
test_word2vec_df = pd.DataFrame(scaled_word2vec_vectors[len(X_train):], columns=[f'w2v_dim_{i}' for i in range(word2vec_model.vector_size)])

train_word2vec_df.insert(0, 'cap_num', train_cap)
train_word2vec_df.insert(0, 'punc_num', train_punc)
train_word2vec_df.insert(0, 'senti_score', train_senti)
train_word2vec_df.insert(0, 'length', train_length)
train_word2vec_df.insert(0, 'is_fake', y_train.reset_index(drop=True))
train_word2vec_df.to_csv('train_w2v.csv')

test_word2vec_df.insert(0, 'cap_num', test_cap)
test_word2vec_df.insert(0, 'punc_num', test_punc)
test_word2vec_df.insert(0, 'senti_score', test_senti)
test_word2vec_df.insert(0, 'length', test_length)
test_word2vec_df.insert(0, 'is_fake', y_test.reset_index(drop=True))
test_word2vec_df.to_csv('test_w2v.csv')
