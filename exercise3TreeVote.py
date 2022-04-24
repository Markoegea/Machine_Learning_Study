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

dataSet = pd.read_csv("BaseDatos/votaciones.csv")
def treeModel():
    noColumns = dataSet.drop(["puedeVotar"],axis=1)
    numerical_cols = [columname for columname in noColumns.columns if 
        noColumns[columname].dtype in ["int64","float64"]]
    x = pd.get_dummies(dataSet[numerical_cols]).values
    y=dataSet["puedeVotar"].values
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.25,random_state=1)
    tree_one = tree.DecisionTreeClassifier()
    tree_one = tree_one.fit(X_train,Y_train)
    tree_one_accuracy=round(tree_one.score(X_test,Y_test),4)
    print("Accuracy: %0.4f"% (tree_one_accuracy))

    dot_data = tree.export_graphviz(tree_one)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('puedeVotarTree.png')
    print("Prediccion Final: \n")
    print(tree_one.predict([[0,17]]))
if __name__ == '__main__':
    treeModel()