import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import euclidean_distances 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import warnings

iris = load_iris()
#Exercice A

#1
df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
Iris = df #exC
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
#Centroid en Jaune
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
"""Lorsque nous effectuons les questions de kmean avec des valeurs projetés ou les valeurs "initials",
les représentations graphiques sont différentes, ainsi que les clusters obtenus.
- Les valeurs projetés savèrent plus présices surtout sur des exemples en particulier mais elles 
peuvent être sujet à moins de modifications et de travaux.
- Le travail sur le jeu de données initial est intéressant quand il faut effectué plusieurs types 
de visualisations et de travaux. Mais il peut s'avérer moins précis et pertinant."""


#Exercice B
#1
def TNN(data, dataf):
    #création de liste vide
    prediction = [] 
    #travail sur un voisin (nous effecturons pas la suite avec des val. diff.)
    voisins=1 
    #On étable la distance existance entre notre jeu de donnée inital et "cible"
    distance = euclidean_distances(data, data)   
    for j in range(distance.shape[0]):        
        distance_e = distance[j] 
        #on retourne les proches voisins avec agsort
        proche_voisin = distance_e.argsort()[:voisins] 
        labels_voisins = [dataf[i] for i in proche_voisin]   
        #predic = np.argmax(labels_voisins, axis=None, out=None) //avec argmax le resultat est nul, l'utilisation de max est plus simple
        predic = max(labels_voisins, key = labels_voisins.count)
        #On met dans notre liste de prediction les valeurs trouvés
        prediction.append(predic) 
    #renvoie les predictions
    prediction = np.array(prediction)
    print(prediction) 
    
#2
#prends en entrée les labels prédit et existant et donne le p. pas prédit  
def TNNE(data, dataf):
	correct = 0
	for i in range(len(data)):
		if data[i] == dataf[i]:
			correct += 1
	return 100 - (correct / float(len(data)) * 100.0)
  
#3
#S.append('setosa')
#TNN(df, S)

#4
warnings.filterwarnings("ignore") #to remove unwanted warnings
x=df.iloc[1:,:3]#features
y=df.iloc[1:,4:]#class labels
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#test_size determines the percentage of test data you want here
#train=80% and test=20% data is randomly split
cv_scores = []
neighbors = list(np.arange(3,50,2))
for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors = n,algorithm = 'brute')
    
    cross_val = cross_val_score(knn,x_train,y_train,cv = 5 , scoring = 'accuracy')
    cv_scores.append(cross_val.mean())
    
error = [1-x for x in cv_scores]
optimal_n = neighbors[ error.index(min(error)) ]
knn_optimal = KNeighborsClassifier(n_neighbors = optimal_n,algorithm = 'brute')
knn_optimal.fit(x_train,y_train)
pred = knn_optimal.predict(x_test)
acc = accuracy_score(y_test,pred)*100
print("Pour k = {0} , la précison est de {1}".format(optimal_n,acc))
"""Lorsqu'on a K=1 les resultats sont similaires et lorsque le nombre de
 voisins augmente la précison augmente"""

#BONUS
def TNNBONUS(data, dataf, voisins=1):
    #création de liste vide
    prediction = [] 
    #travail sur un voisin (nous effecturons pas la suite avec des val. diff.)
    #On étable la distance existance entre notre jeu de donnée inital et "cible"
    distance = euclidean_distances(data, data)   
    for j in range(distance.shape[0]):        
        distance_e = distance[j] 
        #on retourne les proches voisins avec agsort
        proche_voisin = distance_e.argsort()[:voisins] 
        labels_voisins = [dataf[i] for i in proche_voisin]   
        #predic = np.argmax(labels_voisins, axis=None, out=None) //avec argmax le resultat est nul, l'utilisation de max est plus simple
        predic = max(labels_voisins, key = labels_voisins.count)
        #On met dans notre liste de prediction les valeurs trouvés
        prediction.append(predic) 
    #renvoie les predictions
    prediction = np.array(prediction)
    print(prediction)


#Exercice C
#1
def CBN(data, dataf):
    #création de liste vide
    prediction = [] 
    #travail sur un voisin (nous effecturons pas la suite avec des val. diff.)
    voisins=1 
    #On étable la distance existance entre notre jeu de donnée inital et "cible"
    distance = euclidean_distances(data, data)   
    for j in range(distance.shape[0]):        
        distance_e = distance[j] 
        #on retourne les proches voisins avec agsort
        proche_voisin = distance_e.argsort()[:voisins] 
        labels_voisins = [dataf[i] for i in proche_voisin]   
        #predic = np.argmax(labels_voisins, axis=None, out=None) //avec argmax le resultat est nul, l'utilisation de max est plus simple
        predic = max(labels_voisins, key = labels_voisins.count)
        #On met dans notre liste de prediction les valeurs trouvés
        prediction.append(predic) 
    #renvoie les predictions
    prediction = np.array(prediction)
    print(prediction) 
 
#2
def CBNE(data, dataf):
	correct = 0
	for i in range(len(data)):
		if data[i] == dataf[i]:
			correct += 1
	return 100 - (correct / float(len(data)) * 100.0)


#3
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print('Pourcentage de points mal étiquetés et Nombre de points mal étiqueté')
print((X_test.shape[0], (y_test != y_pred).sum()))
#Nombre de points mal étiquetés en % + le nombre de points mal étiqueté



