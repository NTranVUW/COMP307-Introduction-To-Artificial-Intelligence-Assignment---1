import sys
import numpy as np
import math


# returns sum(|x_i-y_i|)
def manhattan(x, y):
    summed = 0
    for i in range(len(x)):
        summed = summed + abs(x[i] - y[i])
    return summed


# returns sqrt(sum((x_i-y_i)^2))
def euclidean(x, y):
    summed = 0
    for i in range(len(x)):
        summed = summed + ((x[i] - y[i]) ** 2)
    return math.sqrt(summed)


# returns either the euclidean or manhattan distance
def distance(x, y, distance_metric):
    if distance_metric == 'manhattan':
        return manhattan(x, y)
    if distance_metric == 'euclidean':
        return euclidean(x, y)
    else:
        raise Exception("distance metric undefined")

# assign instances to the closest centroid
def assign_cluster(data, centroids):
    clusters = [[] for i in range(len(centroids))]
    for i, d in enumerate(data):
        distances = []
        for c in centroids:
            distances.append(distance(d, c, 'euclidean'))
        clusters[np.argmin(distances)].append(i)
    return clusters


# The algorithm:
# 1. Initialize centroids randomly
# 2. Assign each instance to the closest centroid
# 3. Compute the new centroids based on the cluster assignments
# 4. Repeat until convergence (clusters do not change)
def k_means_clustering(data, k):
    centroids = []

    for i in range(k):
        centroids.append(data[np.random.randint(0, len(data))])

    clusters = assign_cluster(data, centroids)
    old_clusters = None
    while clusters != old_clusters:
        old_clusters = clusters
        for i, c in enumerate(centroids):
            centroids[i] = np.mean(data[clusters[i]], axis=0)
        clusters = assign_cluster(data, centroids)
    return clusters, centroids


# normaise data to be between 0 and 1
def min_max_feature_scaling(data):
    min_max_scaling = []
    for i in range(len(data[0])):
        min_max_scaling.append([np.min(data[:, i]), np.max(data[:, i])])
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] = (data[i][j] - min_max_scaling[j][0]) / (min_max_scaling[j][1] - min_max_scaling[j][0])
    return data


def read_data(file):
    with open(file, 'r') as f:
        data = f.readlines()
    data = [line.strip().split()[:-1] for line in data[1:]]  # remove first line and remove last column (the class)
    return np.array(data).astype(float)


if __name__ == '__main__':
    if len(sys.argv) == 5:
        distance_metric = sys.argv[4]
        k = int(sys.argv[3])
        if k < 1:
            raise Exception("k can only be a positive integer greater than 0")
        test = sys.argv[2]
        train = sys.argv[1]
    else:
        raise Exception("usage: python kmeans.py <train_file> <test_file> <k> <distance_metric>")

    # read training data and normalise
    training_data = read_data(train)
    training_data = min_max_feature_scaling(training_data)

    # train the model to find the k centroids
    training_clusters, final_centroids = k_means_clustering(training_data, k)

    # read test data and normalise
    test_data = read_data(test)
    test_data = min_max_feature_scaling(test_data)

    # assign test data to clusters
    test_clusters = assign_cluster(test_data, final_centroids)

    print("Clusters: k = " + str(k) + ", distance metric = " + distance_metric + "\n")
    for i, c in enumerate(test_clusters):
        print("Cluster " + str(i + 1) + ": " + str(c))
        print("Number of instances in cluster " + str(i + 1) + ": " + str(len(c)))
