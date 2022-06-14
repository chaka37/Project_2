import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def read_data(filename):
    data = pd.read_csv()
    data["timestamp"] = pd.to_datetime(data["timestamp"]).dt.date
    data = data.set_index("timestamp")
    return data

data = read_data()

def xy_matrix(xcolumn1, xcolumn2, ycolumn, data):
    X = data[[xcolumn1, xcolumn2]]
    X = X.drop(X.index[1258])
    y = data[[ycolumn]].shift(1)
    y = y.dropna()
    return X, y

X,y = xy_matrix('8EWMA', '20EWMA', 'Signal', data)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test) 


def ann_function(X_train_scaled,X_test_scaled,y_train,y_test):
     ann = tf.keras.models.Sequential()
     ann.add(tf.keras.layers.Dense(units=100,activation="sigmoid"))
     ann.add(tf.keras.layers.Dense(units=50,activation="sigmoid"))
     ann.add(tf.keras.layers.Dense(units=40,activation="tanh"))
     ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
     ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
     ann.fit(X_train_scaled,y_train,batch_size=32,epochs = 500)
     ann_loss, ann_accuracy = ann.evaluate(X_test_scaled, y_test, verbose=2)
     predictions = ann.predict(X_test_scaled)
     return predictions, ann_loss, ann_accuracy


predictions_df = pd.DataFrame(predictions)


def Confusion_matrix():
    cm = confusion_matrix(y_test, predictions)
    cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]
    )
    acc_score = accuracy_score(y_test, predictions)
    return print("Confusion Matrix")
    display(cm_df)
    print(f"Accuracy Score : {acc_score}")
    print("Classification Report")
    print(classification_report(y_test, predictions))
