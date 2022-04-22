from sklearn import cluster
from sklearn.cluster import KMeans
from DataLoader import load_data

def FeatureCluster(n_clusters, data, max_iter=300, tol=0.0001):
    return KMeans(n_clusters=n_clusters, max_iter=max_iter, tol=tol).fit(data)

def FeatureMapping(data, kM:KMeans):
    return kM.predict(data)


if __name__ == '__main__':
    id = "3.11"
    ip = "192.168." + id
    data = load_data("../Data/SYN DoS_pcap%s.csv" % id)
    data = data[:10000]
    
    kM = FeatureCluster(100, data, 1000, tol=0.00001)
    # print(kM.cluster_centers_)