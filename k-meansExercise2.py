import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

dataSet = pd.read_csv("BaseDatos/analisis.csv")
x = np.array(dataSet[["op","ex","ag"]])
y = np.array(dataSet["categoria"])  

def kMeans():
    kmeans = KMeans(n_clusters=4,max_iter=1000).fit(x)
    centroids = kmeans.cluster_centers_
    print(centroids)
    labels = kmeans.predict(x)
    C = kmeans.cluster_centers_
    return kmeans,centroids,labels,C


def thirdDimension():
    colores = ["red","green","blue","cyan"]
    asignar = []
    for row in labels:
        asignar.append(colores[row])
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x[:,0],x[:,1],x[:,2],c=asignar,s=60)
    ax.scatter(C[:,0],C[:,1],C[:,2], marker='*',c=colores,s=1000)
    plt.show()

def secondDimension():
    colores = ["red","green","blue","cyan"]
    asignar = []
    for row in labels:
        asignar.append(colores[row])
    f1 = np.array(pd.DataFrame([dataSet["op"].values,dataSet['ex'].values]))
    f2 = np.array(pd.DataFrame([dataSet["ex"].values,dataSet['ag'].values]))
    def graph(fx,fy,c1,c2):        
        plt.scatter(fx,fy,c=asignar, s=70)
        plt.scatter(C[:,c1],C[:,c2],marker='*',c=colores,s=1000)
        plt.show()
    graph(f1[0],f2[0],0,1)
    graph(f1[0],f2[1],0,2)
    graph(f1[1],f2[1],1,2)

def nearCentroids():
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, x)
    print(closest)
    users =dataSet["usuario"].values
    for row in closest:
        print(users[row])

def predictions():
    x_new = np.array([[45.92,57.74,15.66]])
    new_labels = kmeans.predict(x_new)
    print(new_labels)

def elbowMethod(x):
    wccs = []
    n = 1
    acc = 0
    for i in range (1,11):
        codo = KMeans(n_clusters=i, max_iter=1000, random_state=0)
        codo.fit(x)
        y_kmeans = codo.predict(x)
        wccs.append(codo.inertia_)
        accuracy = round(metrics.adjusted_rand_score(y,y_kmeans),4)
        print(f'Cantidad de Centroides: {i}---Precision: {accuracy}')
        if accuracy > acc:
            acc = accuracy
            n=i
    plt.plot(range(1,11),wccs)
    plt.title("Elbow Method")
    plt.xlabel("Numero de centroides")
    plt.ylabel("WCCS")
    plt.show()
    print(f"Se recomienda emplear una cantidad de {n} centroides, para asi garantizar una precision de {acc}")

if __name__ == '__main__':
    kmeans,centroids,labels,C= kMeans()
    #thirdDimension()
    #secondDimension()
    #nearCentroids()
    predictions()
