import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

str_to_int = {
    'young': 0,
    'pre-presbyopic': 1,
    'presbyopic': 2,
    'myope': 0,
    'hypermetrope': 1,
    'no': 0,
    'yes': 1,
    'reduced': 0,
    'normal': 1,
    'soft': 0,
    'hard': 1,
    'none': 2

}

headers = ['age','specs','astigmatism','tear-rate','lenses']
data = pd.read_csv('lenses.csv')#, header=headers)
data = data.replace(str_to_int).sample(frac=1)
X = data.values[:,:3]
Y = data.values[:,4]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.6, random_state = 100)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

y_pred_en = clf_entropy.predict(X_test)
print ("Accuracy is ", accuracy_score(y_test,y_pred_en) * 100)
