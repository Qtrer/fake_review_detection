import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV

# data processing
train_df = pd.read_csv('train_tfidf.csv')
test_df = pd.read_csv('test_tfidf.csv')
# train_df = pd.read_csv('dealt_train.csv')
# test_df = pd.read_csv('dealt_test.csv')
# train_df = pd.read_csv('train_w2v.csv')
# test_df = pd.read_csv('test_w2v.csv')
# cols = [i for i in train_df.columns if i not in ['cap_num', 'punc_num', 'senti_score', 'length', 'is_fake', '']]
# cols = [i for i in train_df.columns if i not in ['cap_num', 'punc_num', 'is_fake', '']]
# cols = [i for i in train_df.columns if i not in ['senti_score', 'is_fake', '']]
cols = [i for i in train_df.columns if i not in ['is_fake', '']]
train_df2 = train_df[cols]

# X_train, X_test, y_train, y_test = train_test_split(train_df2, train_df['is_fake'], test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(train_df2, train_df['is_fake'], test_size=0.25, random_state=1)
X_test = test_df[cols]
y_test = test_df['is_fake']

# NB
# nb_param_grid = {
#     'alpha': [0.01, 0.1, 0.5, 1.0, 10.0],
# }
# nb_grid_search = GridSearchCV(MultinomialNB(), nb_param_grid, refit=True, verbose=3)
# nb_grid_search.fit(X_train, y_train)
# print(nb_grid_search.best_params_)
#
# nb = MultinomialNB(**nb_grid_search.best_params_)
nb = MultinomialNB(alpha=1)
nb.fit(X_train, y_train)
predictions_nb = nb.predict(X_test)
print("NB Accuracy: ", accuracy_score(y_test, predictions_nb))
print('NB Precision:', precision_score(y_test, predictions_nb, average='macro'))
print('NB Recall:', recall_score(y_test, predictions_nb, average='macro'))
print('NB F1-Score:', f1_score(y_test, predictions_nb, average='macro'))

# NB Accuracy:  0.8375
# NB Precision: 0.8375
# NB Recall: 0.8375527426160339
# NB F1-Score: 0.837493652095785

# SVM
# svm_param_grid = {
#     'C': [0.1, 1, 10, 100, 1000],
#     'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#     'kernel': ['linear', 'rbf']
# }
# svm_grid_search = GridSearchCV(svm.SVC(), svm_param_grid, refit=True, verbose=3)
# svm_grid_search.fit(X_train, y_train)
# print(svm_grid_search.best_params_)
#
# clf = svm.SVC(**svm_grid_search.best_params_)
clf = svm.SVC(C=1, gamma=1, kernel='rbf')
clf.fit(X_train, y_train)
predictions_svm = clf.predict(X_test)
print("SVM Accuracy: ", accuracy_score(y_test, predictions_svm))
print('SVM Precision:', precision_score(y_test, predictions_svm, average='macro'))
print('SVM Recall:', recall_score(y_test, predictions_svm, average='macro'))
print('SVM F1-Score:', f1_score(y_test, predictions_svm, average='macro'))

# SVM Accuracy:  0.8625
# SVM Precision: 0.864313725490196
# SVM Recall: 0.8629473355211752
# SVM F1-Score: 0.8624140087554721

# LR
# lr_param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#     'penalty': ['l1', 'l2'],
#     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# }
# lr_grid_search = GridSearchCV(LogisticRegression(), lr_param_grid, refit=True, verbose=3)
# lr_grid_search.fit(X_train, y_train)
# print(lr_grid_search.best_params_)
#
# lr = LogisticRegression(**lr_grid_search.best_params_)
lr = LogisticRegression(C=10, penalty='l2', solver='newton-cg')
lr.fit(X_train, y_train)
predictions_lr = lr.predict(X_test)
print("LR Accuracy: ", accuracy_score(y_test, predictions_lr))
print('LR Precision:', precision_score(y_test, predictions_lr, average='macro'))
print('LR Recall:', recall_score(y_test, predictions_lr, average='macro'))
print('LR F1-Score:', f1_score(y_test, predictions_lr, average='macro'))

# LR Accuracy:  0.840625
# LR Precision: 0.8462498514204208
# LR Recall: 0.8414205344585092
# LR F1-Score: 0.8401739283720657

# RF
# rf_param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }
# rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_param_grid, refit=True, verbose=3)
# rf_grid_search.fit(X_train, y_train)
# print(rf_grid_search.best_params_)
#
# rf = RandomForestClassifier(**rf_grid_search.best_params_)
rf = RandomForestClassifier(bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=200)
rf.fit(X_train, y_train)
predictions_rf = rf.predict(X_test)
print("RF Accuracy: ", accuracy_score(y_test, predictions_rf))
print('RF Precision:', precision_score(y_test, predictions_rf, average='macro'))
print('RF Recall:', recall_score(y_test, predictions_rf, average='macro'))
print('RF F1-Score:', f1_score(y_test, predictions_rf, average='macro'))
