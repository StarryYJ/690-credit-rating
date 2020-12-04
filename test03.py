from test01 import *


company_start = [0]
for i in range(1, len(data)):
	if data.iloc[i-1, 0] != data.iloc[i, 0]:
		company_start.append(i)


test = company_start
train = list(range(len(data)))
for i in test:
	train.remove(i)


#
clf = OneVsRestClassifier(LogisticRegression(penalty='l2'))
clf.fit(predictors.iloc[train, :], predictand[train])
pred_clf = clf.predict(predictors.iloc[test, :])
cm_clf = confusion_matrix(predictand[test], pred_clf, labels=ratings)
print(cm_clf)
accuracy_clf = sum(np.diag(cm_clf))/len(test)
print(accuracy_clf)
plot_confusion_matrix(cm_clf, title='Multi-class learning with Logistic Regression', classes=ratings)

# Random Forest
rf = RandomForestClassifier()
rf.fit(predictors.iloc[train, :], predictand[train])
pred_rf = rf.predict(predictors.iloc[test, :])
cm_rf = confusion_matrix(predictand[test], pred_rf, labels=ratings)
print(cm_rf)
plot_confusion_matrix(cm_rf, title='Random Forest', classes=ratings)
accuracy_rf = sum(np.diag(cm_rf))/len(test)
print(accuracy_rf)


# Naive Bayes
rf = GaussianNB()
rf.fit(predictors.iloc[train, :], predictand[train])
pred_rf = rf.predict(predictors.iloc[test, :])
cm_rf = confusion_matrix(predictand[test], pred_rf, labels=ratings)
print(cm_rf)
plot_confusion_matrix(cm_rf, title='Naive Bayes', classes=ratings)
accuracy_rf = sum(np.diag(cm_rf))/len(test)
print(accuracy_rf)














