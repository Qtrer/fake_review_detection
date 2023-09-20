import pandas as pd
from sklearn.decomposition import TruncatedSVD

train_df = pd.read_csv('train_tfidf.csv')
test_df = pd.read_csv('test_tfidf.csv')
cols = [i for i in train_df.columns if i not in ['cap_num', 'punc_num', 'senti_score', 'is_fake']]

train_data = train_df[cols]
test_data = test_df[cols]


svd = TruncatedSVD(n_components=100)

# Perform SVD on the TF-IDF vectors
train_vectors_svd = (svd.fit_transform(train_data) + 1) / 2
test_vectors_svd = (svd.transform(test_data) + 1) / 2

dealt_train_df = pd.DataFrame(train_vectors_svd)
dealt_train_df.insert(0, 'cap_num', train_df['cap_num'].reset_index(drop=True))
dealt_train_df.insert(0, 'punc_num', train_df['punc_num'].reset_index(drop=True))
dealt_train_df.insert(0, 'senti_score', train_df['senti_score'].reset_index(drop=True))
dealt_train_df.insert(0, 'is_fake', train_df['is_fake'].reset_index(drop=True))
dealt_test_df = pd.DataFrame(test_vectors_svd)
dealt_test_df.insert(0, 'cap_num', test_df['cap_num'].reset_index(drop=True))
dealt_test_df.insert(0, 'punc_num', test_df['punc_num'].reset_index(drop=True))
dealt_test_df.insert(0, 'senti_score', test_df['senti_score'].reset_index(drop=True))
dealt_test_df.insert(0, 'is_fake', test_df['is_fake'].reset_index(drop=True))
dealt_train_df.to_csv('dealt_train.csv')
dealt_test_df.to_csv('dealt_test.csv')
