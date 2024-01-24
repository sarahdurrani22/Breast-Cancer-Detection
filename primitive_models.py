
# Don't forget to install lazypredict package into your environment
# pip install lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
y= data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)

# Create a LazyClassifier instance
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)

# Fit the classifier on the training data and make predictions on the test data
models, predictions = clf.fit(X_train, X_test, y_train, y_test)