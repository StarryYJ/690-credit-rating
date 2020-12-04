from test01 import *

# like cross validation
companies = list(Counter(data['Global Company Key']).keys())
accuracy_rf_vec = []
for c in companies:
	test = []
	train = []
	for i in range(len(data)):
		if data['Global Company Key'][i] == c:
			test.append(i)
		else:
			train.append(i)

	rf.fit(predictors.iloc[train, :], predictand[train])
	pred_rf = rf.predict(predictors.iloc[test, :])
	cm_rf = confusion_matrix(predictand[test], pred_rf)
	accuracy_rf = sum(np.diag(cm_rf))/len(test)
	accuracy_rf_vec.append(accuracy_rf)
print(accuracy_rf_vec)
np.mean(accuracy_rf_vec)
plt.hist(accuracy_rf_vec, color='darkseagreen', alpha=0.7, rwidth=0.85)
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.show()
plt.close()

#
companies = list(Counter(data['Global Company Key']).keys())
test_ind_2 = []
train_ind_2 = []
for i in range(len(data)):
	if data['Global Company Key'][i] in companies[-4:]:
		test_ind_2.append(i)
	else:
		train_ind_2.append(i)


# Random Forest
rf.fit(predictors.iloc[train_ind_2, :], predictand[train_ind_2])
pred_rf = rf.predict(predictors.iloc[test_ind_2, :])
cm_rf = confusion_matrix(predictand[test_ind_2], pred_rf)
print(cm_rf)
accuracy_rf = sum(np.diag(cm_rf))/len(test_ind_2)
print(accuracy_rf)






















