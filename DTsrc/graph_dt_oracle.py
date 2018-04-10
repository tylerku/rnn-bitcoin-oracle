import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

ITERATIONS = 1000

str_to_int = {
    'buy': 0.0,
    'sell': 1.0,
    'hold': 2.0,
    'None': 0.0
}

headers = ['Timestamp', 'Open', 'High', 'Low', 'Price', 'Volume', '24hr', 'Action']

data = pd.read_csv('../bitcoin_data/bitcoin.csv')#, header=headers)
data = data.replace(str_to_int).sample(frac=1)
data = data.convert_objects(convert_numeric=True)
X = np.nan_to_num(data.values[:,:6])
Y = np.nan_to_num(data.values[:,7])
accuracies = []
testRatios = []
testSizes = []
iterations = []
for i in range(ITERATIONS):
    tr = np.random.rand()
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = tr, random_state = i)
    if (i == 0):
        print(X_train[0])

    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = i,
    max_depth=3, min_samples_leaf=5)

    clf_entropy.fit(X_train, y_train)

    y_pred_en = clf_entropy.predict(X_test)
    acc = accuracy_score(y_test, y_pred_en)
    # print ("Iteration %d: \tTest Ratio: %.2f  \tTest Size: %d   \tAccuracy: %.2f%%" %(i, tr, int((1-tr)*X.shape[0]), acc * 100) )
    accuracies.append(acc)
    testSizes.append(int((1-tr) * X.shape[0]))
    testRatios.append(1-tr)
    iterations.append(i)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

plt.plot(iterations, accuracies)
plt.title("Decision Tree Buy/Hold/Sell Classification Accuracy by Iteration")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.xlim((-1,len(iterations)+1))
plt.ylim((min(accuracies) - .03, max(accuracies) + .03))
plt.show()


plt.scatter(testRatios, accuracies)
plt.title("Decision Tree Buy/Hold/Sell Classifcation Accuracy by Test Set Ratio")
plt.xlabel("Test Set Ratio")
plt.ylabel("Accuracy")
plt.ylim((min(accuracies) - .03, max(accuracies) + .03))
plt.show()


plt.scatter(testSizes, accuracies)
plt.title("Decision Tree Buy/Hold/Sell Classifcation Accuracy by Test Set Size")
plt.xlabel("Test Set Size")
plt.ylabel("Accuracy")
plt.ylim((min(accuracies) - .03, max(accuracies) + .03))
plt.show()