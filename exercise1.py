import random
import numpy as np
from matplotlib import markers, projections
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

dataSet = pd.read_csv("BaseDatos\exercise_1.csv")

def linealModelLearning():
    filtredData = dataSet[(dataSet['Word count'] <= 3500) & (dataSet['# Shares'] <= 80000)]

    colors=['orange','blue']
    tamanios =[30,60]

    f1 = filtredData[['Word count']].values
    f2 = filtredData['# Shares'].values

    asignar=[]
    for index, row in filtredData.iterrows():
        if (row['Word count']>1808):
            asignar.append(colors[0])
        else:
            asignar.append(colors[1])

    
    X_train, X_test, Y_train, Y_test = train_test_split(f1,f2,test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train,Y_train)

    viz_train = plt
    #viz_train.scatter(f1,f2,color=asignar,s=tamanios[0])
    viz_train.scatter(X_train,Y_train,color='brown',s=tamanios[0])
    #viz_train.scatter(X_test, Y_test, color = "purple")
    viz_train.plot(X_train, regressor.predict(X_train),color = "blue")
    viz_train.title("# Palabras vs # Shares")
    viz_train.xlabel("# Words")
    viz_train.ylabel("# Shares")
    viz_train.show()
    print(regressor.score(X_test,Y_test))
    print(regressor.predict([[2000]]))
    print("Valor de la pendiente o coeficione 'a':\n")
    print(regressor.coef_)
    print("Valor de la interseccion 'a':\n")
    print(regressor.intercept_)

def multipleLinealModelLearning():
    suma = suma = (dataSet["# of Links"] + dataSet['# of comments'].fillna(0) + dataSet['# Images video'])
    filtredData = dataSet[(dataSet['Word count'] <= 3500) & (dataSet['# Shares'] <= 80000)]

    f=pd.DataFrame()
    f["Word count"] = filtredData['Word count']
    f["Suma"] = suma
    f1and2 = f
    f3 = filtredData['# Shares'].values
    
    X_train, X_test, Y_train, Y_test = train_test_split(f1and2,f3,test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train,Y_train)

    prediction = regressor.predict(X_train)

    fig = plt.figure()
    viz_train = fig.add_subplot(111,projection='3d')
    xx, yy = np.meshgrid(np.linspace(0, 3500, num=10), np.linspace(0, 60, num=10))
    nuevoX = (regressor.coef_[0] * xx)
    nuevoY = (regressor.coef_[1] * yy)
    z = (nuevoX+nuevoY+regressor.intercept_)
    viz_train.plot_surface(xx, yy, z, alpha=0.2, cmap='hot')
    viz_train.scatter(X_train["Word count"], X_train["Suma"], Y_train, c='blue',s=30)
    viz_train.scatter(X_train["Word count"], X_train["Suma"], prediction, c='red',s=40)
    viz_train.view_init(elev=30., azim=65)
    viz_train.set_xlabel('Cantidad de Palabras')
    viz_train.set_ylabel('Cantidad de Enlaces,Comentarios e Imagenes')
    viz_train.set_zlabel('Compartido en Redes')
    viz_train.set_title('Regresión Lineal con Múltiples Variables')
    plt.show()

    print(regressor.score(X_test,Y_test))
    #print(regressor.predict([[2000]]))
    print("Valor de la pendiente o coeficione 'a':\n")
    print(regressor.coef_)
    print("Valor de la interseccion 'a':\n")
    print(regressor.intercept_)

    print("Prediccion Fina:\n")
    print(regressor.predict([[2000,0+9+2]]))

if __name__ == '__main__':
    #linealModelLearning()
    multipleLinealModelLearning()