#import des librairies l'environnement
import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

#chargement de base de données iris
iris = datasets.load_iris()

"""#affichage des données
print(iris)
print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)"""

#Stocker les données en tant que DataFrame Pandas 
x=pd.DataFrame(iris.data)
# définir les noms de colonnes
x.columns=['Sepal_Length','Sepal_width','Petal_Length','Petal_width']
y=pd.DataFrame(iris.target)
print(y.columns==['Targets'])
