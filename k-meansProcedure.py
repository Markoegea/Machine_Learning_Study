from numpy import column_stack
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X_iris = iris.data
Y_iris = iris.target

def k_meansGraph():
    x = pd.DataFrame(iris.data, columns=["Sepal Length", "Sepal Width","Petal Length","Petal Width"])
    y = pd.DataFrame(iris.target, columns = ["Target"])
    return x,y
    plt.scatter(x["Sepal Length"], x["Sepal Width"],c = "purple")
    plt.xlabel("Sepal Length",fontsize = 10)
    plt.ylabel("Sepal Width", fontsize=10)
    plt.show()

def k_meansModel(x,y):
    model = KMeans(n_clusters=3,max_iter=1000)
    model.fit(x)
    y_labels = model.labels_
    y_kmeans = model.predict(x)
    print("Predicciones: ", y_kmeans)
    accuracy = metrics.adjusted_rand_score(Y_iris, y_kmeans)
    print(accuracy)
    plt.scatter(x["Sepal Length"], x["Sepal Width"], c=y_kmeans,s=30)
    plt.xlabel("Petal Length", fontsize = 10)
    plt.ylabel("Petal Width", fontsize = 10)
    plt.show()

if __name__ == '__main__':
    x,y = k_meansGraph()
    k_meansModel(x,y)