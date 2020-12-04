import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans




iris = load_iris()
#Exercice A

#1
df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
#appelé df et non data

#2
S = []
for i in range(1,150):
    S.append(df['species'][i])
S1 = set(S)
S1 = list(S1)
S1.sort()
#print(df)
#print(df.columns)
df.drop('species',1,inplace=True)
print(S1)
print(df.columns)

#3
X = iris.data
y = iris.target
target_names = iris.target_names
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
colors = ['navy', 'turquoise', 'darkorange']
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.show()

#4
#J'ai préféré décomposé le travail que de directement faire cl=kmeans(data,nbclusters).
iris = datasets.load_iris()
#importer le jeu de données Iris dataset à l'aide du module pandas
x = pd.DataFrame(iris.data)
x.columns = ['Sepal_Length','Sepal_width','Petal_Length','Petal_width']
y = pd.DataFrame(iris.target)
y.columns = ['Targets']
#Création d'un objet K-Means avec un regroupement en 3 clusters (groupes)
model=KMeans(n_clusters=3)
#application du modèle sur notre jeu de données Iris
model.fit(x)


colormap=np.array(['Red','green','blue'])
"""Visualisation du jeu de données sans altération de ce dernier 
afin de mieux le comprendre et commencer à repérer les cluster"""
plt.scatter(x.Petal_Length, x.Petal_width,c=colormap[y.Targets],s=40)
plt.title('Classification réelle')
plt.show()

#Visualisation des clusters formés par K-Means
plt.scatter(x.Petal_Length, x.Petal_width,c=colormap[model.labels_],s=40)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Classification K-means ')
plt.show()

#5
"""Nous remarquons que lorsque nous effectuons le code plusieurs fois, les graphiques différaient 
un peu quant au choix des clusters.Cela vient de la manière dont la fonction K-mean est construite"""

#6
x = df.iloc[:, [1, 2, 3, 4]].values
#Application de k-mean au jeu de donné 
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)
#Visualisation des closters en se basant sur les différentes especes 
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = '-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'virginica')
#Plot
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
plt.legend()

#7


#Exercice B


#CBN (X, Y)


#Exercice C

 

