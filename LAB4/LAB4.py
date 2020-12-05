import numpy as np
import pandas as pd
import seaborn as sns 
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
from math import sqrt
from math import exp
from math import pi

#Exercice A
#1

iris = load_iris()
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
iris = sns.load_dataset('iris')
Species = iris.iloc[:,4] ; colors = Species.astype('category') 
#Visualisation des clusters formés par K-Means
plt.scatter(x.Petal_Length, x.Petal_width,c=colormap[model.labels_],s=40)
#Centroid en Jaune
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('K-mean pour le jeu de données IRIS avec centroids en jaune')
plt.show()


#5
"""Nous remarquons que lorsque nous effectuons le code plusieurs fois, les graphiques différaient 
un peu quant au choix des clusters. Cela vient de la manière dont la fonction K-mean est construite"""

#6
# Tracer le scatter plot
plt.scatter(iris.iloc[:,1],iris.iloc[:,2],c=colors.cat.codes)
plt.title('Représentation graphique des fleurs')
plt.show()
"""Les resultats obtenus sont différents car les méthodes de représentationd graphique sont différents. Il existe néamoins 
toujours le même nombre de clusters relatifs aux trois espèces de fleur"""


#7
"""
ATTENTION LES REPRESENTATIONS ONT ETE FAITES AVEC DF ET LES VALEURS PROJETES MAIS NOUS AVONS LAISSE DF 


Lorsque nous effectuons les questions de kmean avec des valeurs projetés ou les valeurs "initials",
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
    #On étable la distance existance entre notre jeu de donnée inital et "cible"
    distance = euclidean_distances(data, data)   
    for j in range(distance.shape[0]):        
        distance_e = distance[j] 
        #on retourne les proches voisins avec agsort
        proche_voisin = distance_e.argsort()[:1] 
        labels_voisins = [dataf[i] for i in proche_voisin]   
        #predic = np.argmax(labels_voisins, axis=None, out=None) //avec argmax le resultat est nul, l'utilisation de max est plus simple
        predic = max(labels_voisins, key = labels_voisins.count)
        #On met dans notre liste de prediction les valeurs trouvés
        prediction.append(predic) 
    #renvoie les predictions
    prediction = np.array(prediction)
    print(prediction)

#2
def TNNE(data, dataf):
    correct = 0
    prediction = [] 
    distance = euclidean_distances(data, data)   
    for j in range(distance.shape[0]):        
        distance_e = distance[j] 
        proche_voisin = distance_e.argsort()[:1] 
        labels_voisins = [dataf[i] for i in proche_voisin]   
        predic = max(labels_voisins, key = labels_voisins.count)
        prediction.append(predic) 
    prediction = np.array(prediction)
    for i in range(len(dataf)):
        if prediction[i] == dataf[i]:
            correct += 1
    return 100 - (correct / float(len(dataf)) * 100.0)

#3
#print(iris.target)
iris = load_iris()
print(TNNE(iris.data, iris.target))
print(TNN(iris.data, iris.target))
#En utilidsant la fonction TNNE, pour les données df, les résultats sont cohérents  

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
print("le k est fixé aléatoirement, pour en avoir un autre, veuillez relancer le programme")
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
# On casse le jeu de données par valeurs de "class"
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated

# Calcul de la moyenne d'un nb
def mean(numbers):
	return sum(numbers)/float(len(numbers))

# Calculate de la "standard deviation" pour un nombre donnée 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)

# Calculer la moyenne
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries

# Divisez l'ensemble de données par classe, puis calculez les statistiques pour chaque ligne
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries


def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

# les probabilités de prédire chaque classe pour une ligne donnée
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities

# Prediction de la class pour une rangé donnée
def predict(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label

#Algorithme Naive sans l'utilisation de GaussianNB() 
def CBN(data, dataf):
	summarize = summarize_by_class(data)
	predictions = list()
	for row in dataf:
		output = predict(summarize, row)
		predictions.append(output)
	print(predictions)

#2
def CBNE(data, dataf):
    correct = 0
    summarize = summarize_by_class(data)
    predictions = list()
    for row in dataf:
        output = predict(summarize, row)
        predictions.append(output)
    for i in range(len(dataf)):
        if predictions[i] == dataf[i]:
            correct += 1
    return 100 - (correct / float(len(data)) * 100.0)

#3
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print('Pourcentage de points bien classés et Nombre de points mal classés')
print((X_test.shape[0], (y_test != y_pred).sum()))
"""
APRES MES TESTS, J'AI RAJOUTE LA PRECISION AFIN DE MIEUX COMPRENDRE L'ALGORITHME ET D'EFFECTUER DES RECHERCHES COMPLEMENTAIRES 

Apres plusieurs recherches et tests nous nous rendons compte que les resultats obtenus sont similaires 
bien que les métohdes de classifications sont légèrements différentes
"""

#BONUS
# Calculer la fonction de distribution de probabilité gaussienne pour x
def calculate_probabilityG(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

# les probabilités de prédire chaque classe pour une ligne donnée
def calculate_class_probabilitiesG(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probabilityG(row[i], mean, stdev)
	return probabilities

# Prediction de la class pour une rangé donnée
def predictG(summaries, row):
	probabilities = calculate_class_probabilitiesG(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label

#Algorithme Naive sans l'utilisation de GaussianNB() 
def CBNG(data, dataf):
	summarize = summarize_by_class(data)
	predictions = list()
	for row in dataf:
		output = predictG(summarize, row)
		predictions.append(output)
	print(predictions)


