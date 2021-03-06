import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

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

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.6, random_state = 100)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)

clf_entropy.fit(X_train, y_train)

y_pred_en = clf_entropy.predict(X_test)
print ("Accuracy is ", accuracy_score(y_test,y_pred_en) * 100)
