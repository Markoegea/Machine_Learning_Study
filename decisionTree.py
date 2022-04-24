import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pydotplus
import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz/bin/'
sns.set()

dataSetTest = pd.read_csv("BaseDatos/titanic-test.csv")
dataSetTrain = pd.read_csv("BaseDatos/titanic-train.csv")
#dataSetTrain.Sex.value_counts().plot(kind='bar',color = ["b","r"])
#plt.title("Distribucion de sobrevivientes")
#plt.show()

def treeModel():
    label_encoder = preprocessing.LabelEncoder()
    #Quitar espacios vacios en la base de datos para no tener datos nulos
    encoder_sex = label_encoder.fit_transform(dataSetTrain["Sex"])
    dataSetTrain["Age"] = dataSetTrain["Age"].fillna(dataSetTrain["Age"].median())
    dataSetTrain["Embarked"]=dataSetTrain["Embarked"].fillna("S")
    #
    dataSetTest["Age"] = dataSetTrain["Age"].fillna(dataSetTrain["Age"].median())
    dataSetTest["Embarked"]=dataSetTrain["Embarked"].fillna("S")
    #Quitar algunas columnas que nos son incesarias para el arbol
    train_predictor = dataSetTrain.drop(["PassengerId","Survived","Name","Ticket","Cabin"],axis=1)
    #Divido las columnas entre cualitativas y cuantitativas
    categorical_cols = [cname for cname in train_predictor.columns if 
        train_predictor[cname].nunique()<10 and 
        train_predictor[cname].dtype == "object"]
    numerical_cols = [cname for cname in train_predictor.columns if 
    train_predictor[cname].dtype in ["int64","float64"]]
    #Uno los datos en una sola variable
    my_cols = categorical_cols + numerical_cols
    #Solo utilizar las columnas cualitativas y cuantitavas
    train_predictor = train_predictor[my_cols]
    #########################################################################
    #Convertir los datos cualitativos a cuantitativos
    dummy_encoded_train_predictors = pd.get_dummies(train_predictor)
    print( dummy_encoded_train_predictors)
    #Entrenamiento del modelo
    y_target = dataSetTrain["Survived"].values
    x_features_one = dummy_encoded_train_predictors.values
    #Entrenar al arbol para darme las predicciones
    X_train, X_test, Y_train, Y_test = train_test_split(x_features_one,y_target,test_size=0.25,random_state=1)
    tree_one = tree.DecisionTreeClassifier()
    tree_one = tree_one.fit(X_train,Y_train)
    tree_one_accuracy=round(tree_one.score(X_test,Y_test),4)
    print("Accuracy: %0.4f"% (tree_one_accuracy))
    #out = StringIO()
    #tree.export_graphviz(tree_one, out_file=out)
    #graph = graph_from_dot_data(out.getvalue())
    #graph.w
    dot_data = tree.export_graphviz(tree_one,
            impurity=True,
            class_names = ["Muerto", "Vivo"],
            rounded=True,
            filled=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('titanicTree.png')

    


if __name__ == '__main__':
    treeModel()