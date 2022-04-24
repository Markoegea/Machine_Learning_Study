import random
import numpy as np
from matplotlib import markers, projections
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn import metrics
import seaborn as sns

dataSet = pd.read_csv("BaseDatos\SalariosConPaises.csv")
#dataSet = pd.read_csv("salarios.csv")

paises = ['COLOMBIA','USA','SPAIN',"UNITED KINGDOM","CHINA"]
paisesToNum = preprocessing.LabelEncoder()
paisesEncoded=paisesToNum.fit_transform(paises)

def linealModelLearning():
    x = dataSet.iloc[:,0:1].values
    y = dataSet.iloc[:,1].values

    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train,Y_train)

    viz_train = plt
    #viz_train.scatter(X_train, Y_train, color = "green")
    viz_train.scatter(X_test, Y_test, color = "purple")
    viz_train.plot(X_train, regressor.predict(X_train),color = "blue")
    viz_train.title("Salario vs Experiencia")
    viz_train.xlabel("Años Experiencia")
    viz_train.ylabel("Salario")
    viz_train.show()
    print(f'{X_test},{Y_test}')
    print(regressor.score(X_test,Y_test))
    print("Valor de la pendiente o coeficione 'a':\n")
    print(regressor.coef_)
    print("Valor de la interseccion 'a':\n")
    print(regressor.intercept_)

def lMLWithCountries():
    x = dataSet[['Aexperiencia','Paises']]
    y = dataSet.iloc[:,1].values

    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train,Y_train)

    prediction = regressor.predict(X_train)

    fig = plt.figure()
    viz_train = fig.add_subplot(111,projection='3d')
    xx, yy = np.meshgrid(np.linspace(0, 10, num=10), np.linspace(0, 5, num=10))
    nuevoX = (regressor.coef_[0]*xx)
    nuevoY = (regressor.coef_[1]*yy)
    z = (nuevoX+nuevoY+regressor.intercept_)
    viz_train.plot_surface(xx,yy,z,alpha=0.2,cmap="hot")
    viz_train.scatter(X_train["Aexperiencia"],X_train["Paises"],Y_train,c="blue",s=30)
    viz_train.scatter(X_train["Aexperiencia"],X_train["Paises"],prediction,c="red",s=30)
    viz_train.view_init(elev=30.,azim=65)
    viz_train.set_xlabel("Años de experiencia")
    viz_train.set_ylabel("Pais de procedencia")
    viz_train.set_yticks(range(len(paisesEncoded)))
    viz_train.set_yticklabels(paisesToNum.inverse_transform(paisesEncoded))
    viz_train.set_zlabel("Salario")
    viz_train.set_title("Salario segun experiencia y pais")
    plt.show()

    f = pd.DataFrame()
    f["Aexperiencia"]=X_test["Aexperiencia"]
    f["Paises"]=X_test["Paises"]
    f["Sueldo"]=Y_test
    print(f)
    print(regressor.score(X_test,Y_test))
    print("Valor de la pendiente o coeficione 'a':\n")
    print(regressor.coef_)
    print("Valor de la interseccion 'a':\n")
    print(regressor.intercept_)
    print("Prediccion final:\n")
    print(regressor.predict([[4,2]]))

def setCountries():
    #Me crea los paises, los pasa a numero y me los asigna a cada elemento de la base de datos, random
    paises = ['COLOMBIA','USA','SPAIN',"UNITED KINGDOM","CHINA"]
    paisesToNum = preprocessing.LabelEncoder()
    paisesEncoded=paisesToNum.fit_transform(paises)
    paisesDataSet = [random.choice(paisesEncoded) for i in range(len(dataSet))]

    dataSet.insert(2,column="Paises",value=paisesDataSet)
    dataSet.to_csv("SalariosConPaises.csv",index=False)

if __name__ == '__main__':
    #setCountries()
    #linealModelLearning()
    lMLWithCountries()