import pandas as pd

dataSet = pd.read_csv("salarios.csv")
#Borrar columnas de un csv
dataSet.drop(['Unnamed: 0.1','Unnamed: 0'],axis = 1, inplace=True)

#Una lista de numeros normal
series = pd.Series([5,10,15,20,25])
print(series[3])
#Una serie de caracteres, que pasa por pandas
cad = pd.Series(['p','l','a','t','z','i'])
print(cad)
#Un dataframe de pandas usando un lista de una dimension
lista = ['Hola','Mundo','robotico']
df = pd.DataFrame(lista)
print(df)
#Un dataframe de pandas, usando un diccionario como argumento
data = {'Nombre':['Juan','Ana','Jose','Arturo'],'Edad':[25,18,23,27],'Pais':['MX','CO','BR','MX']}
df1 = pd.DataFrame(data)
print(df1)
print(df1[['Nombre','Pais']])
#####Manipulacion archivos
data1 = pd.read_csv('canciones.csv')
#print(data1.head(5))
#artista = data1.artists
#print(artista[5])
#info = data1.iloc[15]
#print(info)
#print(data1.tail())
#print(data1.shape)
#print(data1.columns)
#print(data1["name"].describe())
#print(data1.sort_index(axis = 0, ascending= False))
#####################Reto##############################
filas = pd.DataFrame(data1[90:])
print(filas.sort_values(by="artists",axis = 0, ascending= True))

columnas = data1[["name","artists","duration_ms"]]
print(columnas.sort_values(by="duration_ms",axis = 0, ascending= True))