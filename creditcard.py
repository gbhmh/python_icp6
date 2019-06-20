import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.spatial.distance import cdist

data = pd.read_csv('CC.csv')
# nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False))
# nulls.columns = ['Null Count']
# nulls.index.name  = 'Feature'
# print(nulls)


credit = data.select_dtypes(include=[np.number]).interpolate().dropna()
print(sum(credit.isnull().sum()  != 0))
nulls = pd.DataFrame(credit.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null Count']
nulls.index.name  = 'Feature'
# print(nulls)

# sns.FacetGrid(credit, hue="TENURE", height=4).map(plt.scatter, "CREDIT_LIMIT", "PRC_FULL_PAYMENT").add_legend()
# plt.show()
# g = sns.pairplot(credit, hue="TENURE")
# plt.show()
# print(credit.info())
# sns.set(style="ticks", color_codes=True)
# g = sns.pairplot(credit, hue="cluster")
# plt.show()
# print("hai")
#
x = credit.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
y = credit.iloc[:,-1]
print(x.shape, y.shape)

# M = range(1,16)
# for m in M:
#     plt.scatter(x,x.iloc[:m])
# plt.show()

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
from sklearn import metrics
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)
from sklearn.cluster import KMeans
nclusters = 4 # this is the k in kmeans
seed = 0
K = range(2,10)
for k in K:
    km = KMeans(n_clusters=k, random_state=seed)
    km.fit(X_scaled)
# predict the cluster for each data point
    y_cluster_kmeans = km.predict(X_scaled)

    score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
    print('silhouette score for clusters ' +str(k)+' is : ' + str(score))

Sum_of_squared_distances = []
K = range(1,11)
for k in K:
    km = KMeans(n_clusters=k,init='k-means++',max_iter=300,n_init=10,random_state=0)
    km = km.fit(X_scaled)
    Sum_of_squared_distances.append(km.inertia_)
# Plot the elbow
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('The Elbow Method showing the optimal k')
plt.show()

pca = PCA(n_components=8)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
print("original shape:   ", X_scaled.shape)
print("transformed shape:", X_pca.shape)
X_new = pca.inverse_transform(X_pca)
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.2)
# plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
# plt.axis('equal');
M = range(2,10)
for m in M:
    km = KMeans(n_clusters=m, random_state=seed)
    km.fit(X_pca)
# predict the cluster for each data point
    y_cluster_kmeans_pca = km.predict(X_pca)

    score1 = metrics.silhouette_score(X_pca, y_cluster_kmeans_pca)
    print('silhouette score for clusters after PCA is ' +str(m)+' is : ' + str(score1))

