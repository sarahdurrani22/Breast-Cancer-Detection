import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import dataset_analysis as dataset_analysis
import matplotlib.pyplot as plt

breast_cancer_data_set = dataset_analysis.fetch_dataset_metadata()
data_url = breast_cancer_data_set.metadata.data_url

# Loading dataset into a DataFrame from the given URL
df = pd.read_csv(data_url)
# Display the first few rows of the DataFrame
print(df.head())

# Further processing, training, and evaluation can be done with the 'df' DataFrame

### Removing unwanted columns
df.drop('ID',axis=1,inplace=True)
 
### Converting categorical labels to numerical
df['Diagnosis'] = df['Diagnosis'].map({'M':1,'B':0})
df.head()

### Dividing dataframe into features and labels
y = df["Diagnosis"]
X = df.drop(["Diagnosis"], axis=1)

training_accuracy = ""
testing_accuracy = ""

print(y.shape)
print(X.shape)

"""### Dividing data into train and test split"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

### Reshaping dataset

y_train = y_train.values.reshape(-1, 1).T
y_test = y_test.values.reshape(-1, 1).T
print(y_train.shape)
print(y_test.shape)

X_train_flatten = X_train.values.reshape(X_train.shape[0], -1).T
X_test_flatten = X_test.values.reshape(X_test.shape[0], -1).T

print("X_train Flatten Shape", str(X_train_flatten.shape))
print("X_test Flatten Shape", str(X_test_flatten.shape))


### Normalizing the data

# Machine learning algorithms like linear regression, logistic regression, neural network, etc. that use gradient descent as an optimization technique require data to be scaled.
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

X_train = NormalizeData(X_train_flatten)
X_test = NormalizeData(X_test_flatten)


## Creating custom logistic regression model

### Parameter initialization

# Weight initialization is used to define the initial values for the parameters in neural network models prior to training the models on a dataset.
def initializer(dimension):
    # weights initialization
    w = np.zeros(shape=(dimension, 1))
    # bias initialization
    b = 0

    return w, b

### Activation Function

# sigmoid is general for binary logistic regression
def sigmoidActivation(z):
    sig = 1 / (1 + np.exp(-z))

    return sig

### Calculating Cost and Gradient

def costFunction(w, b, X, Y):
    m = X.shape[1]

    # Calculating Forward Pass
    Act =  sigmoidActivation(np.dot(w.T, X) + b)
    # Computing cost
    cost = (-1 / m) * np.sum(Y * np.log(Act) + (1 - Y) * (np.log(1 - Act)))
    cost = np.squeeze(cost)

    # Calculating Backward Pass for gradients
    dw = (1 / m) * np.dot(X, (Act - Y).T)
    db = (1 / m) * np.sum(Act - Y)
    # Creating dict for gradients
    grads = {"dw" : dw,
             "db" : db}

    return grads, cost

### Optimizer using gradient descent algorithm

def optimizer(w, b, X, Y, epochs, lr):
    # list to store costs
    costs = []
    for i in range(epochs):
        # Computing gradients
        grads, cost = costFunction(w, b, X, Y)
        # Extracting derivatives from grads dictionary
        dw = grads["dw"]
        db = grads["db"]
        # updating parameters
        w = w - lr * dw
        b = b - lr * db
        # Storing and display costs cost after every 100 training records
        if i % 100 == 0:
            costs.append(cost)
            print ("Cost after epoch %i: %f" % (i, cost))
        # Collecting parameters and gradients
        params = {"w": w, "b": b}
        grads = {"dw":dw, "db":db}


    return params, grads, costs

### Predicting the label using learned parameters

def prediction(w, b, X):
    m = X.shape[1]
    # shaping weights according to X set
    w = w.reshape(X.shape[0], 1)
    Label_predict = np.zeros((1, m))
    # Compute the probability
    Act = sigmoidActivation(np.dot(w.T, X) + b)
    # looping over all probabilities
    for i in range(Act.shape[1]):
        # transforming probabilites into actual labels
        Label_predict[0, i] = 1 if Act[0, i] > 0.5 else 0
    return Label_predict

### Combining all the functions to create the logistic regression model
def LogisticRegression(X_train, y_train, X_test, y_test, epochs=3000, lr=0.01):
    # initializing logistic regression parameters
    w, b = initializer(X_train.shape[0])
    # Computing cost and gradients
    params, grads, costs = optimizer(w, b, X_train, y_train, epochs, lr)
    # Extracting parameters from grads dictionary
    w = params["w"]
    b = params["b"]
    # Predicting train and test sets
    Label_predict_train = prediction(w, b, X_train)
    Label_predict_test = prediction(w, b, X_test)
    # Printing train and test set accuracy
    global training_accuracy 
    training_accuracy = "Training accuracy: {} %".format(100 - np.mean(np.abs(Label_predict_train - y_train)) * 100)

    global testing_accuracy
    testing_accuracy = "Testing accuracy: {} %".format(100 - np.mean(np.abs(Label_predict_test - y_test)) * 100)

    print(training_accuracy)
    print(testing_accuracy)

    all_param_dict = {"w":w, "b":b, "costs":costs, "learning_rate":lr, "epochs":epochs}

    return all_param_dict

### Training the model
all_param_dict = LogisticRegression(X_train, y_train, X_test, y_test, epochs=3000, lr=0.005)

# graph plotting
def Graph():
    test_loss = all_param_dict['costs']
    plt.plot(test_loss)
    plt.title("Epoch Test Loss")
    plt.ylabel('costs')
    plt.xlabel('epochs')
    plt.show()

def Accuracy():
    print(training_accuracy)
    print(testing_accuracy)