from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

wines = datasets.load_wine()
X_wines = wines.data
Y_wines = wines.target


def k_meansGraph():
    x = pd.DataFrame(wines.data, columns=wines.feature_names)
    y = pd.DataFrame(wines.target, columns=["Target"])
    scaler = StandardScaler()
    scaler.fit(wines.data)
    x_Scaled = scaler.transform(wines.data)
    return  x_Scaled,y
    plt.scatter(x["alcohol"],x["malic_acid"],x["ash"],c = "green")
    plt.xlabel("alcohol",fontsize=10)
    plt.ylabel("hue",fontsize=10)
    plt.show()
    x.to_csv("Vinos.csv",index=False)

def k_meansModel(x,y):
    model = KMeans(n_clusters=3,max_iter=1000)
    model.fit(x)
    y_kmeans = model.predict(x)
    print("Predicciones: ", y_kmeans)
    accuracy = metrics.adjusted_rand_score(Y_wines,y_kmeans)
    print(accuracy)
    y_kmeans_df = pd.DataFrame(y_kmeans,columns=["Prediction"])
    x_kmeans_df = pd.DataFrame(x)
    z = pd.concat([x_kmeans_df,y_kmeans_df], axis=1)
    sns.pairplot(z,hue="Prediction")
    plt.show()

def elbowMethod(x):
    wccs = []
    n = 1
    acc = 0
    for i in range (1,11):
        codo = KMeans(n_clusters=i, max_iter=1000, random_state=0)
        codo.fit(x)
        y_kmeans = codo.predict(x)
        wccs.append(codo.inertia_)
        accuracy = round(metrics.adjusted_rand_score(Y_wines,y_kmeans),4)
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



if __name__ == "__main__":
    x,y = k_meansGraph()
    #elbowMethod(x)
    k_meansModel(x,y)