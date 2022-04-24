import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

dataSet = pd.read_csv("BaseDatos/artists_billboard.csv")

def treeModel(encoded):
    x = encoded.drop(["top"],axis=1).values
    y = encoded["top"]
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.05,random_state=0)
    decision_tree = tree.DecisionTreeClassifier(criterion="entropy",min_samples_split=20,min_samples_leaf=5,max_depth=5,class_weight={1:3.5})
    decision_tree.fit(X_train,Y_train)
    #Organizar una tabla para poder colocar probar mejor la eficacia del del algoritmo
    sera = pd.DataFrame((X_test),columns=('moodEncoded', 'tempoEncoded', 'genreEncoded','artist_typeEncoded','edadEncoded','durationEncoded'))
    y_Sera = pd.DataFrame(Y_test)
    sera["top"] = y_Sera.values
    #Forma Opcional, obtener achivo png
    with open(r"tree1.dot","w") as f:
        f = tree.export_graphviz(decision_tree,
            out_file=f,
            max_depth=7,
            impurity=True,
            feature_names=list(encoded.drop(["top"],axis=1)),
            class_names = ["No", "N1 Billboard"],
            rounded=True,
            filled=True
        )
    check_call(["dot","-Tpng",r"tree1.dot","-o",r"tree1.png"])
    PImage("tree1.png")
    acc_decision_tree = round(decision_tree.score(X_test,Y_test)*100,2)
    print(acc_decision_tree)
    #Prediccion Final!!!!!
    x_predict = pd.DataFrame(columns=("top",'moodEncoded', 'tempoEncoded', 'genreEncoded','artist_typeEncoded','edadEncoded','durationEncoded'))
    #Modificar datos para obetener prediccion
    x_predict.loc[0]=(0,6,2,4,3,3,0)
    #x_predict.loc[0]=(0,6,2,1,3,2,3)
    y_pred = decision_tree.predict(x_predict.drop(["top"],axis=1))
    print("Prediccion: "+ str(y_pred))
    y_proba = decision_tree.predict_proba(x_predict.drop(["top"], axis=1))
    print("Probabilidad de Acierto: "+str((y_proba[0][y_pred]*100))+"%")
def treeEnconded():
    drop_columns = ["id","title","artist","mood","tempo","genre","artist_type","chart_date","anioNacimiento","durationSeg","edad_en_billboard"]
    dataSet_Encoded = dataSet.drop(drop_columns,axis=1)
    return(dataSet_Encoded)

def treedepth(encoded):
    cv = KFold(n_splits=10) # Numero deseado de "folds" que haremos
    accuracies = list()
    max_attributes = len(list(encoded))
    depth_range = range(1, max_attributes + 1)

    # Testearemos la profundidad de 1 a cantidad de atributos +1
    for depth in depth_range:
        fold_accuracy = []
        tree_model = tree.DecisionTreeClassifier(criterion='entropy',
                                             min_samples_split=20,
                                             min_samples_leaf=5,
                                             max_depth = depth,
                                             class_weight={1:3.5})
        for train_fold, valid_fold in cv.split(encoded):
            f_train = encoded.loc[train_fold] 
            f_valid = encoded.loc[valid_fold] 

            model = tree_model.fit(X = f_train.drop(['top'], axis=1), 
                               y = f_train["top"]) 
            valid_acc = model.score(X = f_valid.drop(['top'], axis=1), 
                                y = f_valid["top"]) # calculamos la precision con el segmento de validacion
            fold_accuracy.append(valid_acc)

        avg = sum(fold_accuracy)/len(fold_accuracy)
        accuracies.append(avg)
    
    # Mostramos los resultados obtenidos
    df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
    df = df[["Max Depth", "Average Accuracy"]]
    print(df.to_string(index=False))

def categorizeData():
    dataSet["moodEncoded"] = dataSet["mood"].map({"Energizing": 6,
        "Empowering":6,
        "Cool":5,
        "Yearning":4,
        "Excited":5,
        "Defiant":3,
        "Sensual":2,
        "Gritty":3,
        "Sophisticated":4,
        "Aggressive":4,
        "Fiery":4,
        "Urgent":3,
        "Rowdy":4,
        "Sentimental":4,
        "Easygoing":1,
        "Melancholy":4,
        "Romantic":2,
        "Peaceful":1,
        "Brooding":4,
        "Upbeat":5,
        "Stirring":5,
        "Lively":5,
        "Other":0,"":0}).astype(int)
    dataSet["tempoEncoded"] = dataSet["tempo"].map({"Fast Tempo":0,
        "Medium Tempo":2,
        "Slow Tempo":1, "":0
    }).astype(int)
    dataSet["genreEncoded"] = dataSet["genre"].map({"Urban":4,
        "Pop":3,
        "Traditional":2,
        "Alternative & Punk":1,
        "Electronica":1,
        "Rock":1,
        "Soundtrack":0,
        "Jazz":0,
        "Other":0,"":0
    }).astype(int)
    dataSet["artist_typeEncoded"] = dataSet["artist_type"].map({
        "Female":2,
        "Male":3,
        "Mixed":1,
        "":0
    }).astype(int)
    dataSet.loc[dataSet["edad_en_billboard"]<=21, "edadEncoded"]=0
    dataSet.loc[dataSet["edad_en_billboard"]>21 & (dataSet["edad_en_billboard"]<=26),"edadEncoded"]=1
    dataSet.loc[dataSet["edad_en_billboard"]>26 & (dataSet["edad_en_billboard"]<=30),"edadEncoded"]=2
    dataSet.loc[dataSet["edad_en_billboard"]>30 & (dataSet["edad_en_billboard"]<=40),"edadEncoded"]=3
    dataSet.loc[dataSet["edad_en_billboard"]>40,"edadEncoded"]=4

    dataSet.loc[dataSet["durationSeg"]<=150, "durationEncoded"]=0
    dataSet.loc[dataSet["durationSeg"]>150 & (dataSet["durationSeg"]<=180),"durationEncoded"]=1
    dataSet.loc[dataSet["durationSeg"]>180 & (dataSet["durationSeg"]<=210),"durationEncoded"]=2
    dataSet.loc[dataSet["durationSeg"]>210 & (dataSet["durationSeg"]<=240),"durationEncoded"]=3
    dataSet.loc[dataSet["durationSeg"]>240 & (dataSet["durationSeg"]<=270),"durationEncoded"]=4
    dataSet.loc[dataSet["durationSeg"]>270 & (dataSet["durationSeg"]<=300),"durationEncoded"]=5
    dataSet.loc[dataSet["durationSeg"]>300,"durationEncoded"]=6

def seeData():
    f1 = dataSet["chart_date"].values
    f2 = dataSet["durationSeg"].values

    colores = ["orange","blue"]
    tamanios = [60,40]
    asignar = []
    asignar2 = []
    for index, row in dataSet.iterrows():
        asignar.append(colores[row["top"]])
        asignar2.append(tamanios[row["top"]])
    plt.scatter(f1,f2, c=asignar)
    plt.axis([20030101,20160101,0,600])
    plt.show()


def calcularEdad():
    dataSet["anioNacimiento"]=dataSet.apply(lambda x: edad_fix(x["anioNacimiento"]),axis=1)
    dataSet["edad_en_billboard"]= dataSet.apply(lambda x: calculaEdad(x["anioNacimiento"],x["chart_date"]), axis=1)
    age_avg = dataSet["edad_en_billboard"].mean()
    age_Std=dataSet["edad_en_billboard"].std()
    age_null_count = dataSet["edad_en_billboard"].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_Std,age_avg+age_Std,size=age_null_count)

    conValoresNulos = np.isnan(dataSet["edad_en_billboard"])
    dataSet.loc[np.isnan(dataSet["edad_en_billboard"]),"edad_en_billboard"] = age_null_random_list
    dataSet["edad_en_billboard"] = dataSet["edad_en_billboard"].astype(int)

def edad_fix(anio):
    if anio==0:
        return None
    return anio

def calculaEdad(anio,cuando):
    cad = str(cuando)
    momento = cad[:4]
    if anio == 0.0:
        return None
    return int(momento)-anio


if __name__ == '__main__':
    calcularEdad()
    categorizeData()
    treedepth(treeEnconded())
    treeModel(treeEnconded())