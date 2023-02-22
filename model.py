from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import preprocess


#=======================load data & remove unnamed columns ==============================#
mails = pd.read_csv("mails.csv", encoding='latin-1')
mails = mails.loc[:, ~mails.columns.str.contains('^Unnamed')]
#=======================load data & remove unnamed columns ==============================#

print(mails.shape)

mails['type'] = mails['type'].map({
    'ham' : 0,
    'spam' : 1
    })

mails['message'] = mails['message'].apply(preprocess.preprocess_message)

X = mails['message']
Y = mails['type']


vectorizer = CountVectorizer()
spam_features = vectorizer.fit_transform(mails['message'])


x_train, x_test, y_train, y_test = train_test_split(spam_features, mails['type'], test_size=0.10)

params = {
    'criterion':  ['gini', 'entropy'],
    'max_depth':  [None, 2, 4, 6, 8, 10],
    'splitter': ['best', 'random']
}

model  = DecisionTreeClassifier()

inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=params, cv=inner_cv)

grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_


model = DecisionTreeClassifier(**best_params)
model.fit(x_train, y_train)


scores = cross_val_score(model, x_test, y_test, cv=outer_cv)
print("scores:", scores)
print("Accuracy (cross_validation): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


y_pred = model.predict(x_test)
print("Confusion matrix:", confusion_matrix(y_test, y_pred))
print("Accuracy test:", accuracy_score(y_test, y_pred))


#================================test with unseen data==================================================#

#=======================load data & remove unnamed columns ==============================#
new_msg = pd.read_csv("new_data.csv", encoding='latin-1')
new_msg = new_msg.loc[:, ~new_msg.columns.str.contains('^Unnamed')]
#=======================load data & remove unnamed columns ==============================#

print(new_msg.shape)

new_msg['message'] = new_msg['message'].apply(preprocess.preprocess_message)

spam_features_new = vectorizer.transform(new_msg['message'])

y_pred_new = model.predict(spam_features_new)

print(y_pred_new)

#================================test with unseen data==================================================#
