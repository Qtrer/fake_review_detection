import pandas as pd
import scipy.io as scio
from keras import Sequential, optimizers
from keras.src import losses
from keras.src.layers import Conv1D, Dense, Dropout, Embedding, GlobalMaxPooling1D, LSTM, SimpleRNN, BatchNormalization
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D
# from tensorflow.keras.layers import LSTM
# from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
import tensorflow as tf

# data processing
train_df = pd.read_csv('train_tfidf.csv')
test_df = pd.read_csv('test_tfidf.csv')
# train_df = pd.read_csv('dealt_train.csv')
# test_df = pd.read_csv('dealt_test.csv')
# train_df = pd.read_csv('train_w2v.csv')
# test_df = pd.read_csv('test_w2v.csv')
# cols = [i for i in train_df.columns if i not in ['cap_num', 'punc_num', 'senti_score', 'is_fake']]
# cols = [i for i in train_df.columns if i not in ['senti_score', 'is_fake']]
cols = [i for i in train_df.columns if i not in ['is_fake']]
train_df2 = train_df[cols]

# X_train, X_test, y_train, y_test = train_test_split(train_df2, train_df['is_fake'], test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(train_df2, train_df['is_fake'], test_size=0.25, random_state=1)
# X_train = train_df[cols]
# y_train = train_df['is_fake']
X_test = test_df[cols]
y_test = test_df['is_fake']

X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val_reshaped = X_val.values.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test_reshaped = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

# feed_forward NN
# model_fnn = Sequential()
# model_fnn.add(Dense(1024, activation='relu', input_shape=(X_train.shape[1], )))
# model_fnn.add(Dropout(0.2))
# model_fnn.add(Dense(512, activation='relu'))
# model_fnn.add(Dropout(0.2))
# model_fnn.add(Dense(256, activation='relu'))
# model_fnn.add(Dropout(0.2))
# model_fnn.add(Dense(1, activation='sigmoid'))
# model_fnn.compile(loss=losses.BinaryCrossentropy(),
#                   optimizer=optimizers.Adam(learning_rate=0.001),
#                   metrics=['accuracy'])
#
# acc = 0
# pre = 0
# rec = 0
# f1s = 0
# for i in range(5):
#     model_fnn.fit(X_train_reshaped, y_train, epochs=30, batch_size=32, validation_data=(X_val_reshaped, y_val))
#     pred_y = model_fnn.predict(X_test_reshaped)
#     pred_y = (pred_y > 0.5)
#     acc += accuracy_score(y_test, pred_y) / 5
#     pre += precision_score(y_test, pred_y, average='macro') / 5
#     rec += recall_score(y_test, pred_y, average='macro') / 5
#     f1s += f1_score(y_test, pred_y, average='macro') / 5
#     print(confusion_matrix(y_test, pred_y))
# print(acc)
# print(pre)
# print(rec)
# print(f1s)

# model_fnn.fit(X_train_reshaped, y_train, epochs=30, batch_size=32, validation_data=(X_val_reshaped, y_val))
# pred_y = model_fnn.predict(X_test_reshaped)
# pred_y = (pred_y > 0.5)
# print("FNN Accuracy: ", accuracy_score(y_test, pred_y))
# print('FNN Precision:', precision_score(y_test, pred_y, average='macro'))
# print('FNN Recall:', recall_score(y_test, pred_y, average='macro'))
# print('FNN F1-Score:', f1_score(y_test, pred_y, average='macro'))
# print("FNN Confusion Matrix: \n", confusion_matrix(y_test, pred_y))

# FNN Accuracy:  0.865625
# FNN Precision: 0.8671382107449351
# FNN Recall: 0.8660337552742616
# FNN F1-Score: 0.8655606686793484

# CNN
# model_cnn = Sequential()
# model_cnn.add(Conv1D(512, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
# model_cnn.add(BatchNormalization())
# model_cnn.add(GlobalMaxPooling1D())
# model_cnn.add(Dense(256, activation='relu'))
# model_cnn.add(Dropout(0.2))
# model_cnn.add(Dense(1, activation='sigmoid'))
# model_cnn.compile(loss='binary_crossentropy',
#                   optimizer=optimizers.Adam(learning_rate=0.01),
#                   metrics=['accuracy'])
#
# acc = 0
# pre = 0
# rec = 0
# f1s = 0
# for i in range(5):
#     model_cnn.fit(X_train_reshaped, y_train, epochs=30, batch_size=32, validation_data=(X_val_reshaped, y_val))
#     pred_y = model_cnn.predict(X_test_reshaped)
#     pred_y = (pred_y > 0.5)
#     acc += accuracy_score(y_test, pred_y) / 5
#     pre += precision_score(y_test, pred_y, average='macro') / 5
#     rec += recall_score(y_test, pred_y, average='macro') / 5
#     f1s += f1_score(y_test, pred_y, average='macro') / 5
#     print(confusion_matrix(y_test, pred_y))
# print(acc)
# print(pre)
# print(rec)
# print(f1s)

# model_cnn.fit(X_train_reshaped, y_train.values, epochs=30, batch_size=32, validation_data=(X_val_reshaped, y_val))
# predictions_cnn = model_cnn.predict(X_test_reshaped)
# predictions_cnn = (predictions_cnn > 0.5)
# print("CNN Accuracy: ", accuracy_score(y_test, predictions_cnn))
# print('CNN Precision:', precision_score(y_test, predictions_cnn, average='macro'))
# print('CNN Recall:', recall_score(y_test, predictions_cnn, average='macro'))
# print('CNN F1-Score:', f1_score(y_test, predictions_cnn, average='macro'))
# print("CNN Confusion Matrix: \n", confusion_matrix(y_test, predictions_cnn))

# CNN Accuracy:  0.58125
# CNN Precision: 0.607036374478235
# CNN Recall: 0.5841537740271918
# CNN F1-Score: 0.5589203423304805


# RNN
# model_rnn = Sequential()
# model_rnn.add(SimpleRNN(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(X_train.shape[1], 1)))
# model_rnn.add(BatchNormalization())
# model_rnn.add(Dense(1, activation='sigmoid'))
# model_rnn.compile(loss='binary_crossentropy',
#                   optimizer=optimizers.Adam(learning_rate=0.01),
#                   metrics=['accuracy'])
#
# acc = 0
# pre = 0
# rec = 0
# f1s = 0
# for i in range(5):
#     model_rnn.fit(X_train_reshaped, y_train, epochs=30, batch_size=32, validation_data=(X_val_reshaped, y_val))
#     pred_y = model_rnn.predict(X_test_reshaped)
#     pred_y = (pred_y > 0.5)
#     acc += accuracy_score(y_test, pred_y) / 5
#     pre += precision_score(y_test, pred_y, average='macro') / 5
#     rec += recall_score(y_test, pred_y, average='macro') / 5
#     f1s += f1_score(y_test, pred_y, average='macro') / 5
#     print(confusion_matrix(y_test, pred_y))
# print(acc)
# print(pre)
# print(rec)
# print(f1s)

# model_rnn.fit(X_train_reshaped, y_train, epochs=30, batch_size=32, validation_data=(X_val_reshaped, y_val))
# predictions_rnn = model_rnn.predict(X_test_reshaped)
# predictions_rnn = (predictions_rnn > 0.5)
# print("RNN Accuracy: ", accuracy_score(y_test, predictions_rnn))
# print('RNN Precision:', precision_score(y_test, predictions_rnn, average='macro'))
# print('RNN Recall:', recall_score(y_test, predictions_rnn, average='macro'))
# print('RNN F1-Score:', f1_score(y_test, predictions_rnn, average='macro'))
# print("RNN Confusion Matrix: \n", confusion_matrix(y_test, predictions_rnn))

# LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(X_train.shape[1], 1)))
model_lstm.add(BatchNormalization())
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(loss='binary_crossentropy',
                   optimizer=optimizers.Adam(learning_rate=0.01),
                   metrics=['accuracy'])

acc = 0
pre = 0
rec = 0
f1s = 0
for i in range(5):
    model_lstm.fit(X_train_reshaped, y_train, epochs=30, batch_size=32, validation_data=(X_val_reshaped, y_val))
    pred_y = model_lstm.predict(X_test_reshaped)
    pred_y = (pred_y > 0.5)
    acc += accuracy_score(y_test, pred_y) / 5
    pre += precision_score(y_test, pred_y, average='macro') / 5
    rec += recall_score(y_test, pred_y, average='macro') / 5
    f1s += f1_score(y_test, pred_y, average='macro') / 5
    print(confusion_matrix(y_test, pred_y))
print(acc)
print(pre)
print(rec)
print(f1s)

# model_lstm.fit(X_train_reshaped, y_train, epochs=30, batch_size=32, validation_data=(X_val_reshaped, y_val))
# predictions_lstm = model_lstm.predict(X_test_reshaped)
# predictions_lstm = (predictions_lstm > 0.5)
# print("LSTM Accuracy: ", accuracy_score(y_test, predictions_lstm))
# print('LSTM Precision:', precision_score(y_test, predictions_lstm, average='macro'))
# print('LSTM Recall:', recall_score(y_test, predictions_lstm, average='macro'))
# print('LSTM F1-Score:', f1_score(y_test, predictions_lstm, average='macro'))
# print("LSTM Confusion Matrix: \n", confusion_matrix(y_test, predictions_lstm))
