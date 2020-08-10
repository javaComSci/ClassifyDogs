import numpy as np
import math
import pickle

class KMeans():
    def __init__(self, filename, num_clusters = 3):
        self.data = np.load(filename)
        # self.data = np.asmatrix([[1,2,3], [4,5,6]])
        self.num_clusters = num_clusters
    

    # with the data, do the clustering
    def cluster(self): 
        # get the examples and the features associated with the data
        num_examples = self.data.shape[0]
        num_features = self.data.shape[1]

        # create inital cluster points
        cluster_points = np.random.rand(self.num_clusters, num_features) * 2
        # print(cluster_points.shape)

        # do iterations to find the final cluster points
        for k in range(300):
            if k % 10 == 0:
                print(k)

            # keep track of the clusters that it belongs to
            clusters_belonging = np.zeros(num_examples)

            # find the cluster that each point is closest to
            for example in range(num_examples):
                distances = np.sqrt(np.sum(np.square(cluster_points - self.data[example]), axis = 1))
                clusters_belonging[example] = np.argmin(distances, axis=0)

            # find the new clustering
            for i in range(self.num_clusters):
                points_in_cluster = 0
                new_mid = None
                for j in range(num_examples):
                    if clusters_belonging[j] == i and new_mid is None:
                        new_mid = self.data[j, :]
                        points_in_cluster += 1
                    elif clusters_belonging[j] == i:
                        new_mid = self.data[j, :] + new_mid
                        points_in_cluster += 1
                if points_in_cluster == 0:
                    continue
                else:
                    cluster_points[i, :] = new_mid/points_in_cluster

        # final clusters
        print(cluster_points)
        print(clusters_belonging)
        
