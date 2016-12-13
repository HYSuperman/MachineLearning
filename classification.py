import sys
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

category=['Film & Animation', 'Music', 'Entertainment', 'Sports', 'Travel & Places', 
	'News & Politics', 'Autos & Vehicles', 'People & Blogs', 'Comedy', 'Pets & Animals', 
	'Howto & DIY', 'Gadgets & Games']

# 10%, 30%, 50%, 70%, 90%

views = [224, 951, 2511, 6200, 19818]
lengths = [44, 106, 186, 258, 470]
comments = [1, 3, 6, 11, 32]
ratings = [1, 4, 8, 17, 48]

xdataset = []

ydataset = []

def getCate(text):
	return category.index(text)

def get_rate(text):
	rate = float(text)
	for i in range(10):
		if rate < i*0.5:
			return i
	return 10

def getviews(text):
	view = int(text)
	for i in range(5):
		if view < views[i]:
			return i
	return 5

def getCom(text):
	comment = int(text)
	for i in range(5):
		if comment < comments[i]:
			return i
	return 5

def getLen(text):
	length = int(text)
	for i in range(5):
		if length < lengths[i]:
			return i
	return 5

def getRat(text):
	rating = int(text)
	for i in range(5):
		if rating < ratings[i]:
			return i
	return 5

with open('sum.txt', 'r') as f:

	for line in f:
		texts = line[:-1].split('\t')
		if int(getCom(texts[8])) >= 0:
			data = [getCate(texts[3]), getLen(texts[4]), getviews(texts[5]), getRat(texts[7]), getCom(texts[8])]
			xdataset.append(data)
			rate = get_rate(texts[6])
			ydataset.append(rate)
		else:
			pass

# X = np.array(xdataset)
# Y = np.array(ydataset)

# with open('mid.txt', 'w') as w:
# 	for s in X:
# 		w.write(str(s)+'\n')

X_train, X_test, y_train, y_test = train_test_split(xdataset, ydataset, test_size = 0.2, random_state=23)

#-------Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

predictions = gnb.predict(X_test)
result = []
tp = 0
for i in range(len(y_test)):
	if y_test[i] == predictions[i]:
		result.append(1)
		tp += 1
	else:
		result.append(0)

print(len(result))
print(tp)

#-------Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

predictions = mnb.predict(X_test)
result = []
tp = 0
for i in range(len(y_test)):
	if y_test[i] == predictions[i]:
		result.append(1)
		tp += 1
	else:
		result.append(0)

print(len(result))
print(tp)

gnb = LogisticRegression(C=10)
gnb.fit(X_train, y_train)
predictions = gnb.predict(X_test)
result = []
tp = 0
for i in range(len(y_test)):
	if y_test[i] == predictions[i]:
		result.append(1)
		tp += 1
	else:
		result.append(0)

print(len(result))
print(tp)

gnb = RandomForestClassifier()
gnb.fit(X_train, y_train)
predictions = gnb.predict(X_test)
result = []
tp = 0
for i in range(len(y_test)):
	if y_test[i] == predictions[i]:
		result.append(1)
		tp += 1
	else:
		result.append(0)

print(len(result))
print(tp)

gnb = DecisionTreeClassifier()
gnb.fit(X_train, y_train)
predictions = gnb.predict(X_test)
result = []
tp = 0
for i in range(len(y_test)):
	if y_test[i] == predictions[i]:
		result.append(1)
		tp += 1
	else:
		result.append(0)

print(len(result))
print(tp)