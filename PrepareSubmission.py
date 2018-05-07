import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics.classification import log_loss
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.linear_model import RidgeClassifier
from sklearn import svm
from skbayes.linear_models import EBLogisticRegression, VBLogisticRegression
import warnings
warnings.filterwarnings('ignore')


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def train_nb(data, labels, test_set):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer(use_idf=False)),
                         ('clf', MultinomialNB(alpha=0.001)),
                         ])

    text_clf.fit(data, labels)

    return text_clf.predict(data), text_clf.predict(test_set)


def train_svc(data, labels, test_set):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer(use_idf=False)),
                         ('clf', LinearSVC()),
                         ])

    text_clf.fit(data, labels)

    return text_clf.predict(data), text_clf.predict(test_set)


def train_nb_cat(data, labels, test_set):
    count_vect = CountVectorizer(stop_words='english')
    X_train_counts = count_vect.fit_transform(data)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # naive bayes
    clf = MultinomialNB()
    clf.fit(X_train_tfidf, labels)

    X_new_counts = count_vect.transform(test_set)
    transformed_test_set = tfidf_transformer.transform(X_new_counts)

    predicted_train = clf.predict_proba(X_train_tfidf)
    predicted_test = clf.predict_proba(transformed_test_set)

    return predicted_train, predicted_test


def train_text_classifier(data, labels, test_set):

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer(use_idf=False)),
                         ('clf', RidgeClassifier()),
                         ])

    text_clf.fit(data, labels)

    return text_clf.predict(data), text_clf.predict(test_set)


# read data
#df = pd.read_csv('data/treated_train.csv')
df = pd.read_csv('data/train_data1.CSV.csv')
dft = pd.read_csv('data/test_data.csv')
df_full = df.append(dft)
#df_full = pd.read_csv('data/train_data.csv')
#df = df_full.head(int(0.8 * len(df_full)))
#dft = df.tail(int(0.2 * len(df_full)))

################################################################################
# pre process
################################################################################
# discretise category
category_le = preprocessing.LabelEncoder()
category_le.fit(df['category'])
# df['disc_category'] = category_le.transform(df['category'])
df['disc_category'] = category_le.transform(df['category'])


#df['combone'] = df['description'] + ' ' + df['merchant']

#df.loc[df['description'] == 'MANUAL', 'description'] = df.loc[df['description'] == 'MANUAL', 'combone']


# change string into datetime
df_full['transactionDate'] = pd.to_datetime(df_full['transactionDate'])
df_full['time'] = df_full['transactionDate'].apply(lambda x: x.hour * 60 + x.minute)
df['transactionDate'] = pd.to_datetime(df['transactionDate'])
df['time'] = df['transactionDate'].apply(lambda x: x.hour * 60 + x.minute)
dft['transactionDate'] = pd.to_datetime(dft['transactionDate'])
dft['time'] = dft['transactionDate'].apply(lambda x: x.hour * 60 + x.minute)

# engineer day of the week
df_full['day'] = df_full['transactionDate'].apply(lambda x: x.day)
df_full['weekday'] = df_full['day'].apply(lambda x: 1 if x == 1 or x == 2 or x in range(5, 9) else 0)
df_full['weekend'] = df_full['day'].apply(lambda x: 1 if x == 3 or x == 4 or x == 10 else 0)

df['day'] = df['transactionDate'].apply(lambda x: x.day)
df['weekday'] = df['day'].apply(lambda x: 1 if x == 1 or x == 2 or x in range(5, 9) else 0)
df['weekend'] = df['day'].apply(lambda x: 1 if x == 3 or x == 4 or x == 10 else 0)

dft['day'] = dft['transactionDate'].apply(lambda x: x.day)
dft['weekday'] = dft['day'].apply(lambda x: 1 if x == 1 or x == 2 or x in range(5, 9) else 0)
dft['weekend'] = dft['day'].apply(lambda x: 1 if x == 3 or x == 4 or x == 10 else 0)

# remove negative sign
df_full['amount'] = df_full['amount'].apply(lambda x: -x)
df['amount'] = df['amount'].apply(lambda x: -x)
dft['amount'] = dft['amount'].apply(lambda x: -x)

# add rounded variable and dummy
df_full['rounded_true'] = df_full['amount'].apply(lambda x: np.ceil(x % 1))
df_full['rounded_false'] = df_full['amount'].apply(lambda x: np.floor(x % 1))
df['rounded_true'] = df['amount'].apply(lambda x: np.ceil(x % 1))
df['rounded_false'] = df['amount'].apply(lambda x: np.floor(x % 1))
dft['rounded_true'] = dft['amount'].apply(lambda x: np.ceil(x % 1))
dft['rounded_false'] = dft['amount'].apply(lambda x: np.floor(x % 1))


time_bin = 24
amount_bin = 15
df_full['time_bin'] = pd.cut(df_full['time'],
                             time_bin,
                             labels=range(0, time_bin)).astype('int')

df_full['amount_bin'] = pd.cut(df_full['amount'],
                               amount_bin,
                               labels=range(0, amount_bin)).astype('int')

df['time_bin'] = pd.cut(df['time'],
                        time_bin,
                        labels=range(0, time_bin)).astype('int')

df['amount_bin'] = pd.cut(df['amount'],
                          amount_bin,
                          labels=range(0, amount_bin)).astype('int')

dft['time_bin'] = pd.cut(dft['time'],
                         time_bin,
                         labels=range(0, time_bin)).astype('int')

dft['amount_bin'] = pd.cut(dft['amount'],
                           amount_bin,
                           labels=range(0, amount_bin)).astype('int')

# no amount scaled

################################################################################
# process text
################################################################################
# added some parameters

# print(df)

################################################################################
# process text
################################################################################

# merchant
#print("Adding merchant features")
################################################################################
# add merchant features to train set
merchant_nb_train, merchant_nb_test = train_nb_cat(df['merchant'], df['disc_category'], dft['merchant'])

col_list_naive_merchant = []
for i in range(len(category_le.classes_)):
    col_list_naive_merchant.append('naive_merchant_' + str(i))
    df = df.assign(**{'naive_merchant_' + str(i): merchant_nb_train[:, i]})
    dft = dft.assign(**{'naive_merchant_' + str(i): merchant_nb_test[:, i]})

merchant_nb_train, merchant_nb_test = train_nb(df['merchant'], df['disc_category'], dft['merchant'])
df['naive_merchant'] = merchant_nb_train
dft['naive_merchant'] = merchant_nb_test

# add merchant features to test set
merchant_tc_train, merchant_tc_test = train_text_classifier(df['merchant'], df['disc_category'], dft['merchant'])

df['merchant_tc'] = merchant_tc_train
dft['merchant_tc'] = merchant_tc_test

merchant_tc_train, merchant_tc_test = train_svc(df['merchant'], df['disc_category'], dft['merchant'])

df['merchant_svc'] = merchant_tc_train
dft['merchant_svc'] = merchant_tc_test

# description
#print("Adding description features")
################################################################################
# add description features to train set
description_nb_train, description_nb_test = train_nb_cat(df['description'], df['disc_category'], dft['description'])

col_list_naive_description = []
for i in range(len(category_le.classes_)):
    col_list_naive_description.append('naive_description_' + str(i))
    df = df.assign(**{'naive_description_' + str(i): description_nb_train[:, i]})
    dft = dft.assign(**{'naive_description_' + str(i): description_nb_test[:, i]})

description_tc_train, description_tc_test = train_text_classifier(df['description'], df['disc_category'], dft['description'])
df['description_tc'] = description_tc_train
dft['description_tc'] = description_tc_test

description_tc_train, description_tc_test = train_svc(df['description'], df['disc_category'], dft['description'])
df['description_svc'] = description_tc_train
dft['description_svc'] = description_tc_test

# prediction only
description_nb_train, description_nb_test = train_nb(df['description'], df['disc_category'], dft['description'])

df['naive_description'] = description_nb_train
dft['naive_description'] = description_nb_test


################################################################################
# Adding combined features
################################################################################

data = df['merchant'] + " " + df['description']
test_data = dft['merchant'] + " " + dft['description']

'''
combo_nb_train, combo_nb_test = train_nb_cat(data, df['disc_category'], test_data)

col_list_naive_combo = []
for i in range(len(category_le.classes_)):
    col_list_naive_description.append('naive_combo_' + str(i))
    df = df.assign(**{'naive_combo_' + str(i): combo_nb_train[:, i]})
    dft = dft.assign(**{'naive_combo_' + str(i): combo_nb_test[:, i]})
    '''

combo_nb_train, combo_nb_test = train_nb(data, df['disc_category'], test_data)
df['naive_combo'] = combo_nb_train
dft['naive_combo'] = combo_nb_test

combo_tc_train, combo_tc_test = train_text_classifier(data, df['disc_category'], test_data)
df['combo_tc'] = combo_tc_train
dft['combo_tc'] = combo_tc_test

combo_tc_train, combo_tc_test = train_svc(data, df['disc_category'], test_data)
df['combo_svc'] = combo_tc_train
dft['combo_svc'] = combo_tc_test


################################################################################
# Bayesian Logistic regression meta features
################################################################################

# BAYESIAN
feature_cols = [  # 'code',
    #'amount',
    #'amount_scaled',
    #'rounded_true',  # 'rounded_false',
    #'time',
    'weekday',  # 'weekend',  # 'friday', 'saturday', 'sunday',

    # cant take time bin because only exists in full df
    #'time_bin', 'amount_bin',
    #'naive_merchant',
    #'naive_description',
    #'naive_combo',
    'merchant_tc',
    'description_tc',
    'combo_tc',
    #'merchant_svc',
    #'description_svc',
]

cat_feature_cols = [  # 'rounded_true',
    #'weekday',
    #'time_bin', 'amount_bin',
    #'naive_description',
    #'naive_merchant', 'naive_combo',
    'merchant_tc',
    'description_tc',
    'combo_tc',
    #'merchant_svc',
    #'description_svc',
]


X = df.loc[:, feature_cols]
X_test = dft.loc[:, feature_cols]
y = df['disc_category']

for col in cat_feature_cols:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# code
X = pd.get_dummies(X)

# time bin
X_test = pd.get_dummies(X_test)
inter_cat = df_full['time_bin'].astype('category')
inter_cat = pd.get_dummies(inter_cat)
p1 = pd.DataFrame(inter_cat.iloc[range(len(df))], index=X.index.values)
result = pd.concat([X, p1], axis=1, ignore_index=True)
p2 = pd.DataFrame(inter_cat.iloc[range(len(df), len(df_full))], index=X_test.index.values)
result1 = pd.concat([X_test, p2], axis=1, ignore_index=True)
# amount bin
X_test = pd.get_dummies(X_test)
inter_cat = df_full['amount_bin'].astype('category')
inter_cat = pd.get_dummies(inter_cat)
p1 = pd.DataFrame(inter_cat.iloc[range(len(df))], index=X.index.values)
result = pd.concat([X, p1], axis=1, ignore_index=True)
p2 = pd.DataFrame(inter_cat.iloc[range(len(df), len(df_full))], index=X_test.index.values)
result1 = pd.concat([X_test, p2], axis=1, ignore_index=True)

# codes
X_test = pd.get_dummies(X_test)
inter_cat = df_full['code'].astype('category')
inter_cat = pd.get_dummies(inter_cat)
p1 = pd.DataFrame(inter_cat.iloc[range(len(df))], index=X.index.values)
result = pd.concat([X, p1], axis=1, ignore_index=True)
p2 = pd.DataFrame(inter_cat.iloc[range(len(df), len(df_full))], index=X_test.index.values)
result1 = pd.concat([X_test, p2], axis=1, ignore_index=True)


# Bayesian

# train
eblr = EBLogisticRegression(tol_solver=1e-3)
vblr = VBLogisticRegression()
# eblr.fit(result,y)
vblr.fit(result, y)

# print("Laplace score", eblr.score(result1, y_true))
#print("VB score", vblr.score(result1, y_true))

# add to test dataframes
predicted = vblr.predict_proba(result)
predicted1 = vblr.predict_proba(result1)

col_list_naive_combo = []
for i in range(len(category_le.classes_)):
    col_list_naive_description.append('naive_combo_' + str(i))
    df = df.assign(**{'naive_combo_' + str(i): predicted[:, i]})
    dft = dft.assign(**{'naive_combo_' + str(i): predicted1[:, i]})

################################################################################
# Logistic regression meta feature
################################################################################

feature_cols = [  # 'code',
    #'amount',
    #'amount_scaled',
    #'rounded_true',  # 'rounded_false',
    #'time',
    'weekday',  # 'weekend',  # 'friday', 'saturday', 'sunday',

    # cant take time bin because only exists in full df
    #'time_bin', 'amount_bin',
    #'naive_merchant',
    #'naive_description',
    #'naive_combo',
    'merchant_tc',
    'description_tc',
    'combo_tc',
    #'merchant_svc',
    #'description_svc',
]

cat_feature_cols = [  # 'rounded_true',
    #'weekday',
    #'time_bin', 'amount_bin',
    #'naive_description',
    #'naive_merchant', 'naive_combo',
    'merchant_tc',
    'description_tc',
    'combo_tc',
    #'merchant_svc',
    #'description_svc',
]

for i in col_list_naive_combo:
    feature_cols.append(i)

X = df.loc[:, feature_cols]
X_test = dft.loc[:, feature_cols]
y = df['disc_category']

for col in cat_feature_cols:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# code
X = pd.get_dummies(X)

# time bin
X_test = pd.get_dummies(X_test)
inter_cat = df_full['time_bin'].astype('category')
inter_cat = pd.get_dummies(inter_cat)
p1 = pd.DataFrame(inter_cat.iloc[range(len(df))], index=X.index.values)
result = pd.concat([X, p1], axis=1, ignore_index=True)
p2 = pd.DataFrame(inter_cat.iloc[range(len(df), len(df_full))], index=X_test.index.values)
result1 = pd.concat([X_test, p2], axis=1, ignore_index=True)
# amount bin
X_test = pd.get_dummies(X_test)
inter_cat = df_full['amount_bin'].astype('category')
inter_cat = pd.get_dummies(inter_cat)
p1 = pd.DataFrame(inter_cat.iloc[range(len(df))], index=X.index.values)
result = pd.concat([X, p1], axis=1, ignore_index=True)
p2 = pd.DataFrame(inter_cat.iloc[range(len(df), len(df_full))], index=X_test.index.values)
result1 = pd.concat([X_test, p2], axis=1, ignore_index=True)

# codes
X_test = pd.get_dummies(X_test)
inter_cat = df_full['code'].astype('category')
inter_cat = pd.get_dummies(inter_cat)
p1 = pd.DataFrame(inter_cat.iloc[range(len(df))], index=X.index.values)
result = pd.concat([X, p1], axis=1, ignore_index=True)
p2 = pd.DataFrame(inter_cat.iloc[range(len(df), len(df_full))], index=X_test.index.values)
result1 = pd.concat([X_test, p2], axis=1, ignore_index=True)

# train
print("Fitting Logistic model...")
logreg = linear_model.LogisticRegression(C=1e5, penalty='l1', multi_class='ovr')
logreg.fit(result, y)


# add to test dataframes
print("Adding as meta features...")
predicted = logreg.predict(result)
df['logreg'] = predicted

predicted = logreg.predict(result1)
dft['logreg'] = predicted

#####################################################################
#           add knn
#####################################################################
feature_cols = ['code',  # 'amount', 'rounded_true',  # 'rounded_false',
                #'time', 'weekday',  # 'weekend',  # 'friday', 'saturday', 'sunday',
                #'time_bin', 'amount_bin',
                'naive_merchant',
                'naive_description',
                'naive_combo',
                'merchant_tc',
                'description_tc',
                #'merchant_svc',
                #'description_svc',
                #'combo_tc'
                ]

# for i in col_list_naive_merchant:
#    feature_cols.append(i)
# for i in col_list_naive_description:
#    feature_cols.append(i)
# for i in col_list_naive_combo:
#    feature_cols.append(i)

X = df.loc[:, feature_cols]
y = df['disc_category']
test = dft.loc[:, feature_cols]

feature_cols = ['code',  # 'amount', 'rounded_true',  # 'rounded_false',
                #'time',
                #'weekday',  # 'weekend',  # 'friday', 'saturday', 'sunday',
                #'time_bin', 'amount_bin',
                'naive_merchant', 'naive_description',
                'naive_combo',
                'merchant_tc',
                'description_tc',
                'merchant_svc',
                'description_svc',
                #'combo_tc'
                ]

X1 = df.loc[:, feature_cols]
y1 = df['disc_category']
test1 = dft.loc[:, feature_cols]

#Y = df_full['disc_category']

#############################################################################
# Declare classifier
# for j in range(3):
knn1 = KNeighborsClassifier(n_neighbors=3)
knn1.fit(X, y)


print("Triple threat")
knn2 = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn2.fit(X1, y1)
print("Adding as meta features...")
predicted = knn1.predict(X)
df['knn1'] = predicted
predicted = knn1.predict(test)
dft['knn1'] = predicted

predicted = knn2.predict(X1)
df['knn2'] = predicted
predicted = knn2.predict(test1)
dft['knn2'] = predicted


################################################################################
# Train intermediate XGB
################################################################################
feature_cols = ['code', 'amount', 'rounded_true',  # 'rounded_false',
                'time', 'weekday',  # 'weekend',  # 'friday', 'saturday', 'sunday',
                'time_bin', 'amount_bin',
                'naive_merchant', 'naive_description',
                'naive_combo',
                'merchant_tc',
                'description_tc',
                'combo_tc'
                #'merchant_svc',
                #'description_svc',
                'logreg',
                #'svm',
                #'knn1',
                #'knn2'
                ]
'''
for i in col_list_naive_merchant:
    feature_cols.append(i)
for i in col_list_naive_description:
    feature_cols.append(i)
for i in col_list_naive_combo:
    feature_cols.append(i)
'''


for i in col_list_naive_combo:
    feature_cols.append(i)

X = df.loc[:, feature_cols]
y = df['disc_category']
test = dft.loc[:, feature_cols]

#############################################################################
# Declare classifier
xgb = XGBClassifier(learning_rate=0.02, max_depth=4, n_estimators=500, n_thread=8,
                    nthread=8)
#############################################################################

# Here we go
start_time = timer(None)  # timing starts from this point for "start_time" variable
print("Training the XGB model...")
xgb.fit(X, y)
timer(start_time)  # timing ends here for "start_time" variable

# add to test dataframes
predicted = xgb.predict(X)
df['xgb'] = predicted

predictions = xgb.predict(test)
dft['xgb'] = predictions

################################################################################
# Get in usable format
################################################################################
feature_cols = ['code', 'amount', 'rounded_true',  # 'rounded_false',
                'time', 'weekday',  # 'weekend',  # 'friday', 'saturday', 'sunday',
                'time_bin', 'amount_bin',
                #'naive_merchant', 'naive_description',
                #'naive_combo',
                'merchant_tc',
                'description_tc',
                #'combo_tc'
                #'merchant_svc',
                #'description_svc',
                'logreg',
                #'svm',
                'knn1',
                #'knn2',
                'xgb'
                ]
'''
for i in col_list_naive_merchant:
    feature_cols.append(i)
for i in col_list_naive_description:
    feature_cols.append(i)
for i in col_list_naive_combo:
    feature_cols.append(i)
'''


for i in col_list_naive_combo:
    feature_cols.append(i)

X = df.loc[:, feature_cols]
y = df['disc_category']
test = dft.loc[:, feature_cols]

#############################################################################
# Declare classifier
xgb = XGBClassifier(learning_rate=0.02, max_depth=4, n_estimators=500, n_thread=8,
                    nthread=8)
#############################################################################

# Here we go
start_time = timer(None)  # timing starts from this point for "start_time" variable
# random_search.fit(X, Y)
print("Training the XGB model...")
xgb.fit(X, y)
timer(start_time)  # timing ends here for "start_time" variable
print("Making predictions...")
predictions = xgb.predict(test)


# predict
print("Making final predictions...")
clf_probs = xgb.predict_proba(test)
print("Train shape", X.shape)
print("Test shape", test.shape)

# set up predictions dataframe
category_le = preprocessing.LabelEncoder()
df = pd.read_csv('data/train_data.csv')
category_le.fit(df['category'])
submission_data = pd.DataFrame(columns=category_le.classes_, data=clf_probs)
print("Submission data")
print(submission_data.head(3))

# get id column
dft = pd.read_csv('data/test_data.csv')
submit_id = dft.take([0], axis=1)

# write results to csv
result = pd.concat([submit_id, submission_data], axis=1)
result.to_csv('Stack.csv', index=False)
