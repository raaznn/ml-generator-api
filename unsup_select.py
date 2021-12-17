from sklearn.cluster import KMeans
def unsupervised_model(df):
  kmeans = KMeans(n_clusters=3, random_state=0).fit(df)
  return kmeans,kmeans.score,kmeans.score