import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import KerasClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
import os

print("start")

def load_sgrid_data():

    data = pd.read_csv(directory+"/sgrid/in/smart_grid_stability_augmented.csv")
    map1 = {'unstable': 0, 'stable': 1}
    data['stabf'] = data['stabf'].replace(map1)

    #divide data
    X = data.iloc[:, :13]
    y = data.iloc[:, 13]
    #print(X.head)

    X_training = X.iloc[:54000, :]
    y_training = y.iloc[:54000]

    X_testing = X.iloc[54000:, :]
    y_testing = y.iloc[54000:]

    #SMALL RUNTIME:
    #X_training = X.iloc[:10000, :]
    #y_training = y.iloc[:10000]

    #X_testing = X.iloc[10000:15000, :]
    #y_testing = y.iloc[10000:15000:]

    X_training = X_training.values
    y_training = y_training.values

    X_testing = X_testing.values
    y_testing = y_testing.values

    return X_training, y_training, X_testing, y_testing




directory = os.getcwd()
# Step 1: Load the Smart Grid dataset
x_train, y_train, x_test, y_test = load_sgrid_data() #these are numpy arrays

# Step 2: Create the model

# ANN initialization
model = Sequential()
# Input layer and first hidden layer
model.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))
# Second hidden layer
model.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
# Third hidden layer
model.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
# Single-node output layer
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# ANN compilation
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Step 3-4: Create & Train the ART classifier
classifier = KerasClassifier(model=model, use_logits=False)
classifier.fit(x_train, y_train)

# Step 5: Evaluate the ART classifier on benign test examples

y_pred = classifier.predict(x_test)
y_pred[y_pred <= 0.5] = 0
y_pred[y_pred > 0.5] = 1

cm = pd.DataFrame(data=confusion_matrix(y_test, y_pred, labels=[0, 1]),
                  index=["Actual Unstable", "Actual Stable"],
                  columns=["Predicted Unstable", "Predicted Stable"])
print(cm)

print(f'Accuracy per the confusion matrix: {((cm.iloc[0, 0] + cm.iloc[1, 1]) / len(y_test) * 100):.2f}%')


# Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=.2)
x_test_adv = attack.generate(x=x_test)

# Step 7: Evaluate the ART classifier on adversarial test examples

y_pred = classifier.predict(x_test_adv)
y_pred[y_pred <= 0.5] = 0
y_pred[y_pred > 0.5] = 1

cm = pd.DataFrame(data=confusion_matrix(y_test, y_pred, labels=[0, 1]),
                  index=["Actual Unstable", "Actual Stable"],
                  columns=["Predicted Unstable", "Predicted Stable"])
print(cm)

print(f'Accuracy per the confusion matrix: {((cm.iloc[0, 0] + cm.iloc[1, 1]) / len(y_test) * 100):.2f}%')


#SAVE TEST DATASETS TO UNDERSTAND THE CHANGE BY FGSM
pd.DataFrame(x_test).to_csv(directory+"/sgrid/outs/original_test_data.csv")
pd.DataFrame(x_test_adv).to_csv(directory+"/sgrid/outs/FGSM_adv_test_data.csv")


###########################

# Step 6: Generate adversarial test examples
attack = ProjectedGradientDescent(classifier, targeted=False, max_iter=10, eps_step=1, eps=.2)
x_test_adv = attack.generate(x=x_test)

# Step 7: Evaluate the ART classifier on adversarial test examples

y_pred = classifier.predict(x_test_adv)
y_pred[y_pred <= 0.5] = 0
y_pred[y_pred > 0.5] = 1

cm = pd.DataFrame(data=confusion_matrix(y_test, y_pred, labels=[0, 1]),
                  index=["Actual Unstable", "Actual Stable"],
                  columns=["Predicted Unstable", "Predicted Stable"])
print(cm)

print(f'Accuracy per the confusion matrix: {((cm.iloc[0, 0] + cm.iloc[1, 1]) / len(y_test) * 100):.2f}%')


#SAVE TEST DATASETS TO UNDERSTAND THE CHANGE BY PGD
pd.DataFrame(x_test_adv).to_csv(directory+"/sgrid/outs/PGD_adv_test_data.csv")