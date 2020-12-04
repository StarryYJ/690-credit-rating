from test01 import *


# credit chang detect
predictors = data.drop(columns=qualitative).drop(columns=['Global Company Key',
														  'S&P Domestic Long Term Issuer Credit Rating',
														  'Credit Situation Change'])
predictand = data['Credit Situation Change']


# Random Forest
accuracy_rf_vec = []
cm_vec = []
for j in range(100):
	test_ind = random.sample(list(range(len(data))), 400)
	train_ind = list(range(len(data)))
	for i in test_ind:
		train_ind.remove(i)
	rf.fit(predictors.iloc[train_ind, :], predictand[train_ind])
	pred_rf = rf.predict(predictors.iloc[test_ind, :])
	cm_rf = confusion_matrix(predictand[test_ind], pred_rf, labels=[0, 1, 2])
	cm_vec.append(cm_rf)
	accuracy_rf = sum(np.diag(cm_rf))/len(test_ind)
	accuracy_rf_vec.append(accuracy_rf)

np.mean(accuracy_rf_vec)
plot_confusion_matrix(cm_rf)


# Naive Bayes
nb.fit(predictors.iloc[train_ind, :], predictand[train_ind])
pred_nb = rf.predict(predictors.iloc[test_ind, :])
cm_nb = confusion_matrix(predictand[test_ind], pred_nb, labels=[0, 1, 2])
plot_confusion_matrix(cm_nb)
sum(np.diag(cm_rf))/len(test_ind)


accuracy_nb_vec = []
cm_nb_vec = []
for j in range(100):
	test_ind = random.sample(list(range(len(data))), 400)
	train_ind = list(range(len(data)))
	for i in test_ind:
		train_ind.remove(i)
	nb.fit(predictors.iloc[train_ind, :], predictand[train_ind])
	pred_nb = rf.predict(predictors.iloc[test_ind, :])
	cm_nb = confusion_matrix(predictand[test_ind], pred_nb, labels=[0, 1, 2])
	cm_nb_vec.append(cm_nb)
	accuracy_nb = sum(np.diag(cm_nb))/len(test_ind)
	accuracy_nb_vec.append(accuracy_nb)

np.mean(accuracy_nb_vec)

plot_confusion_matrix(cm_nb_vec[99])




