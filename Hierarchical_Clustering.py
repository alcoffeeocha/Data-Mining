# Hierarchical Clustering
# Muhammad Alkahfi Khuzaimy Abdullah (1301174048)

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from pylab import rcParams
import seaborn as sb
import sklearn.metrics as sm

np.set_printoptions(precision=4, suppress=True)
plt.figure(figsize=(10,3))
plt.style.use('seaborn-whitegrid')

# Import Dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Dendogram to find optional number of clusters
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'), truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True)
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')

plt.axhline(y=500)
plt.axhline(y=150)
plt.show()

# Fiting hierarchical clustering to dataset
ac = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_ac = ac.fit_predict(X)

# Visualize the cluster
plt.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_ac == 2, 0], X[y_ac == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_ac == 3, 0], X[y_ac == 3, 1], s = 100, c = 'black', label = 'Careless')
plt.scatter(X[y_ac == 4, 0], X[y_ac == 4, 1], s = 100, c = 'grey', label = 'Sensible')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# 


