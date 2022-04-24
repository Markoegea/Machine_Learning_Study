import matplotlib
import pandas as pd
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

dataSet = pd.read_csv("BaseDatos/usuarios.csv")
#0 - Windows: 86, 1 - Macintosh: 40, 2 - Linux: 44
#print(dataSet.groupby("clase").size())
    #Visualizar datos
    #dataSet.drop(["clase"],1).hist()
    #plt.show()
def logisticRegression():
    x = np.array(dataSet.drop(["clase"],1))
    y = np.array(dataSet["clase"])
    X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)

    model = linear_model.LogisticRegression(max_iter=1000)
    model.fit(X_train,Y_train)
    prediction = model.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(Y_test,prediction)
    print("Exactitud",metrics.accuracy_score(Y_test,prediction))

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

    print(metrics.classification_report(Y_test,prediction))
    X_new = pd.DataFrame({"duracion":[1000], "paginas":[3],"acciones":[20],"valor":[160]})
    print("Prediccion Final: ",model.predict(X_new))

if __name__ == '__main__':
    logisticRegression()