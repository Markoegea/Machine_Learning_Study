import numpy as np

if __name__ == '__main__':
    np.array([10,20,24,5,15,50])
    a = np.array([10,20,24,5,15,50,40,23,34,100])
    #Coger un elemento en especifico segun el indice
    print(a[4])
    #Coger de un indice en adelante
    print(a[5:])
    #Coger de un indice al otro indice los elementos comprendido entre ellos
    print(a[5:7])
    #Coger los elementos desde un indice saltandose n indices
    print(a[0::7])
    #Me creo un arreglo con tantos indices como le indique, todos valiento 0
    np.zeros(5)
    #Me crea un arreglo de dos dimensiones con tantos elementos como le indique, todos valiendo 1
    b=np.ones((4,5))
    print(type(b))
    #Me crea un arreglo de intervalo de 3 a 10 de 5 elementos
    print(np.linspace(3,10,8))
    #Me crea un arreglo de dos dimensiones
    c=np.array([['x','y','z'],['a','b','c']])
    print(c)
    print(type(c))
    #Saber cuantas dimensiones tiene un arreglo
    print(a.ndim)
    #Sort me permite ordenar de mayor a menor un arreglo
    d = [12,4,10,40,2]
    print(np.sort(d))
    #Me permite organizar los datos de un arreglo tal como en una tabla de excel y con sort los ordeno
    cabeceras = [('nombre','S10'),('edad',int)]
    datos =[('Juan',10),('Maria',70),('Javier',42),('Samuel',15)]
    usuarios = np.array(datos,dtype=cabeceras)
    print(np.sort(usuarios,order = 'edad'))
    #Me crea un arreglo con los elementos de 0 a n (parametro en numero que yo le indique)
    print(np.arange(5))
    #
    np.arange(5)
    #
    np.arange(5,50,5)
    #Arreglo bidimensional
    print(np.full((3,5),10))
    #Arreglo bidimension con los valores que yo le pida pero en una diagonal
    e = np.diag([1,2,3,4,5])
    print(e)
    #################Reto#######################
    generos=["femenino","masculino"]
    nombres=[('nombre',list),('genero',list),('estatura',float)]
    personas=[("Silvia",generos[0],1.6),("Aquiles",generos[1],1.8),("Ajax",generos[1],1.85),("Pandora",generos[0],1.57)]
    datos=np.array(personas,dtype=nombres)
    print(np.sort(datos,order='estatura'))

