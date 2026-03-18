import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


#import dataset
df = pd.read_csv('penguins.csv')
# #print penguin type count 
# print(df['species'].nunique()) # 3 types of penguins
#clean NaNs
clean_NaN_df = df.dropna()
#drop cols-> (later) cluster the penguins based on "bill_length_mm" and "bill_depth_mm" 
X = clean_NaN_df[["bill_length_mm", "bill_depth_mm"]].values

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# K-means clustering model (cluster the penguins' types based on "bill_length_mm" and "bill_depth_mm")------------------------------------------------
kmeans = KMeans(n_clusters = 3, random_state = 0)
clusters = kmeans.fit_predict(X)
print(kmeans.cluster_centers_.shape)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Visualize the clusters in an XY plane ("centroids" of each cluster in figure)-----------------------------------------------------------------------
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='viridis')
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    color='red',
    marker='.',
    s=200
)
plt.xlabel("Bill Length")
plt.ylabel("Bill Depth")
plt.title("Penguin Clusters")
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Model Evaluation & Accuracy-------------------------------------------------------------------------------------------------------------------------
#performance evaluation using silhouette score (measures how similar an object is to its own cluster compared to other clusters)
score = silhouette_score(X, clusters)
print("Silhouette Score:", score) # 0.5 is a good score, 1 is perfect clustering, -1 is bad clustering
#compare the clusters with real species labels
# row = cluster data
# col = real data
y = clean_NaN_df['species']
print(pd.crosstab(clusters, y))