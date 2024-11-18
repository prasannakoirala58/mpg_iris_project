# clustering/kmeans_clustering.py
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd

def kmeans_clustering():
    """
    Perform K-means clustering on the Iris dataset (k=4).
    Print cluster sizes for each group.
    """
    iris = sns.load_dataset("iris")
    X = iris.drop(columns=['species'])

    kmeans = KMeans(n_clusters=4, random_state=42)
    iris['cluster'] = kmeans.fit_predict(X)
    
    print("\n### K-means Clustering Results ###")
    print("Cluster Sizes:")
    print(iris['cluster'].value_counts())
    print("Cluster Centers:")
    print(kmeans.cluster_centers_)

if __name__ == "__main__":
    kmeans_clustering()
