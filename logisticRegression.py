import matplotlib
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

dataSet = pd.read_csv("BaseDatos\diabetes.csv")

def logisticModel():
    feature_cols = ["Pregnancies","Insulin","BMI","Age","Glucose","BloodPressure","DiabetesPedigreeFunction"]
    x = dataSet[feature_cols]
    y = dataSet.Outcome
    X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.25,random_state=0)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train,Y_train)
    y_predict = logreg.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(Y_test,y_predict)

    class_names = [0,1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks,class_names)
    plt.yticks(tick_marks,class_names)

    sns.heatmap(pd.DataFrame(cnf_matrix),annot = True,cmap="Greens",fmt = "g")
    plt.tight_layout()
    plt.title("Matriz de confusion")
    plt.ylabel("Etiqueta Actual")
    plt.xlabel("Etiqueta de prediccion")
    plt.show()
    print("Exactitud",metrics.accuracy_score(Y_test,y_predict))


if __name__ == '__main__':
    logisticModel()