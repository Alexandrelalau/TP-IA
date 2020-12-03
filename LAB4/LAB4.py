import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


iris = load_iris()


#Exercice 1

#1
df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


df_iris = df

#2
S = []
for i in range(1,150):
    S.append(df['species'][i])
S1 = set(S)
S1 = list(S1)
S1.sort()
print(S1)
print(df.columns)
df.drop('species',1,inplace=True)
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
#Visualisation du jeu de données sans altération de ce dernier (affichage des fleurs selon leur étiquettes)
plt.scatter(x.Petal_Length, x.Petal_width,c=colormap[y.Targets],s=40)
plt.title('Classification réelle')
plt.show()

#Visualisation des clusters formés par K-Means
plt.scatter(x.Petal_Length, x.Petal_width,c=colormap[model.labels_],s=40)
plt.title('Classification K-means ')
plt.show()

