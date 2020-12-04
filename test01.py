import numpy as np
import random
from sklearn.multiclass import OneVsRestClassifier
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from functions import *
import matplotlib.pyplot as plt
from plot_comfusion_mat import plot_confusion_matrix


data = pd.read_csv('SPRatings.csv').iloc[:, 1:]
data.describe()

test_ind = random.sample(list(range(len(data))), 400)
train_ind = list(range(len(data)))
for i in test_ind:
	train_ind.remove(i)

train_set = data.drop(index=test_ind)
test_set = data.iloc[test_ind, :]


# missing value
missing_detect(data)

# sort data
data = data.sort_values(['Global Company Key', 'Data Date'], ascending=True)

# data type
qualitative, quantitative = quality_quantity_classify(data.drop(columns='S&P Domestic Long Term Issuer Credit Rating'))
predictors = data.drop(columns=qualitative).drop(columns=['Global Company Key',
														  'S&P Domestic Long Term Issuer Credit Rating'])
predictand = data['S&P Domestic Long Term Issuer Credit Rating']

# new column
change_detect = [0]
for i in range(1, len(data)-1):
	if data.iloc[i, 0] == data.iloc[i+1, 0] and data.iloc[i+1, 1] != data.iloc[i, 1]:
		change_detect.append(1)
	elif data.iloc[i, 0] == data.iloc[i+1, 0] and data.iloc[i, 1] != data.iloc[i-1, 1]:
		change_detect.append(2)
	else:
		change_detect.append(0)
change_detect.append(0)
data['Credit Situation Change'] = change_detect


ratings = Counter(data.iloc[:, 1]).keys()
ratings = sorted(ratings)


# fit model
# Multi-class learning with Logistic Regression
clf = OneVsRestClassifier(LogisticRegression(penalty='l2'))
clf.fit(predictors.iloc[train_ind, :], predictand[train_ind])
pred_clf = clf.predict(predictors.iloc[test_ind, :])
cm_clf = confusion_matrix(predictand[test_ind], pred_clf, labels=ratings)
print(cm_clf)
accuracy_clf = sum(np.diag(cm_clf))/len(test_ind)
print(accuracy_clf)
plot_confusion_matrix(cm_clf, title='Multi-class learning with Logistic Regression', classes=ratings)

# Naive Bayes
nb = GaussianNB()
nb.fit(predictors.iloc[train_ind, :], predictand[train_ind])
pred_nb = nb.predict(predictors.iloc[test_ind, :])
cm_nb = confusion_matrix(predictand[test_ind], pred_nb, labels=ratings)
print(cm_nb)
accuracy_nb = sum(np.diag(cm_nb))/len(test_ind)
print(accuracy_nb)

plot_confusion_matrix(cm_nb, title='Naive Bayes', classes=ratings)


# Random Forest
rf = RandomForestClassifier(n_estimators=200)
rf.fit(predictors.iloc[train_ind, :], predictand[train_ind])
pred_rf = rf.predict(predictors.iloc[test_ind, :])
cm_rf = confusion_matrix(predictand[test_ind], pred_rf, labels=ratings)
print(cm_rf)
accuracy_rf = sum(np.diag(cm_rf))/len(test_ind)
print(accuracy_rf)

plot_confusion_matrix(cm_rf, title='Random Forest', classes=ratings)


# future selection
# PCA
pca = PCA(n_components='mle', copy=True)
pca.fit(predictors)
print(pca.explained_variance_ratio_)

pca = PCA(n_components=1, copy=True)
pca_predictors = pca.fit_transform(predictors)
print(pca.explained_variance_ratio_)

# Wrapper(RFE)
Wrapper = RFE(estimator=LogisticRegression(), step=1).fit(predictors, predictand)
print(Wrapper.ranking_)
sum(Wrapper.ranking_ == 1)


# fit(PCA)
glm = LogisticRegression()
glm.fit(pca_predictors[train_ind], np.array(change_detect)[train_ind])
pred_glm = glm.predict(pca_predictors[test_ind])
cm_rf = confusion_matrix(np.array(change_detect)[test_ind], pred_glm, labels=[0, 1, 2])
print(cm_rf)


glm = GaussianNB()
glm.fit(pca_predictors[train_ind], predictand[train_ind])
pred_glm = glm.predict(pca_predictors[test_ind])
cm_rf = confusion_matrix(predictand[test_ind], pred_glm, labels=ratings)
print(cm_rf)
accuracy = sum(np.diag(cm_rf))/len(test_ind)
print(accuracy)
plot_confusion_matrix(cm_rf, title='Naive Bayes', classes=ratings)


glm = RandomForestClassifier()
glm.fit(pca_predictors[train_ind], predictand[train_ind])
pred_glm = glm.predict(pca_predictors[test_ind])
cm_rf = confusion_matrix(predictand[test_ind], pred_glm, labels=ratings)
print(cm_rf)
accuracy = sum(np.diag(cm_rf))/len(test_ind)
print(accuracy)
plot_confusion_matrix(cm_rf, title='Random Forest', classes=ratings)


# fit(wrapper)
col_ind = np.array(range(len(Wrapper.support_)))[Wrapper.support_]

# fit model
# Multi-class learning with Logistic Regression
clf = OneVsRestClassifier(LogisticRegression(penalty='l2'))
clf.fit(predictors.iloc[train_ind, col_ind], predictand[train_ind])
pred_clf = clf.predict(predictors.iloc[test_ind, col_ind])
cm_clf = confusion_matrix(predictand[test_ind], pred_clf, labels=ratings)
print(cm_clf)
accuracy_clf = sum(np.diag(cm_clf))/len(test_ind)
print(accuracy_clf)
plot_confusion_matrix(cm_clf, title='Multi-class learning with Logistic Regression', classes=ratings)

# Naive Bayes
nb = GaussianNB()
nb.fit(predictors.iloc[train_ind, col_ind], predictand[train_ind])
pred_nb = nb.predict(predictors.iloc[test_ind, col_ind])
cm_nb = confusion_matrix(predictand[test_ind], pred_nb, labels=ratings)
print(cm_nb)
accuracy_nb = sum(np.diag(cm_nb))/len(test_ind)
print(accuracy_nb)

plot_confusion_matrix(cm_nb, title='Naive Bayes', classes=ratings)


# Random Forest
rf = RandomForestClassifier(n_estimators=200)
rf.fit(predictors.iloc[train_ind, col_ind], predictand[train_ind])
pred_rf = rf.predict(predictors.iloc[test_ind, col_ind])
cm_rf = confusion_matrix(predictand[test_ind], pred_rf, labels=ratings)
print(cm_rf)
accuracy_rf = sum(np.diag(cm_rf))/len(test_ind)
print(accuracy_rf)

plot_confusion_matrix(cm_rf, title='Random Forest', classes=ratings)


