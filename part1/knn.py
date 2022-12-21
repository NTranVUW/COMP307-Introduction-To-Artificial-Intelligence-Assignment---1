import sys
import math
import operator


# helper function to take the list of nearest neighbours and returns the predicted class
def predict(nearest_neighbours):
    classes = {}
    for nn in nearest_neighbours:
        class_label = nn[1]
        if class_label in classes:
            classes[class_label] = classes[class_label] + 1
        else:
            classes[class_label] = 1
    # sorts a dict of {class: occurrences} by highest occurrences to lowest and returning the first element
    # (the class with the highest occurrences)
    return next(iter(dict(sorted(classes.items(), key=operator.itemgetter(1), reverse=True))))


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


# for each instance in the test set
#   for each instance in the training set
#       compute distance between test instance and training instance
#   sort distances lowest to highest
#   find k lowest values
#   find most common class of k-lowest values
def knn(test_X, test_y, train_X, train_y, k, distance_metric):
    n = 0
    for i, a in enumerate(test_X):
        test_class = test_y[i]
        distances = []
        for j, b in enumerate(train_X):
            train_class = train_y[j]
            distances.append((distance(a, b, distance_metric), train_class))
        distances.sort()
        nearest_neighbours = distances[:k]
        predicted_class = predict(nearest_neighbours)
        print(i, ": Predicted Class: ", predicted_class, ", Actual Class: ", test_class)
        if predicted_class == test_class:
            n = n + 1
    return n


def min_max_feature_scaling(data, max_val, min_val):
    new_data = []
    for d in data:
        new_d = []
        for i, j in enumerate(d):
            new_i = (float(j) - min_val[i]) / (max_val[i] - min_val[i])
            new_d.append(new_i)
        new_data.append(new_d)
    return new_data


# returns a vector of the min/max of each feature
def get_max_min(data):
    max_val = []
    min_val = []
    for i in range(len(data[0])):
        max_val.append(float(max(data, key=lambda x: float(x[i]))[i]))
        min_val.append(float(min(data, key=lambda x: float(x[i]))[i]))
    return max_val, min_val


def read_data(data):
    X = []
    y = []
    with open(data, 'r') as f:
        for line in f:
            line = line.split()
            X.append(line[:-1])
            y.append(line[-1])
    return X[1:], y[1:]  # remove feature names


if __name__ == '__main__':
    if len(sys.argv) == 5:
        distance_metric = sys.argv[4]
        k = int(sys.argv[3])
        if k < 1:
            raise Exception("k can only be a positive integer greater than 0")
        test = sys.argv[2]
        train = sys.argv[1]
    else:
        raise Exception("Usage: python knn.py <training data> <test data> <k> <distance metric>")

    # Pipeline: read data -> scale data -> test on training set -> test on test set
    training_X, training_y = read_data(train)
    train_max, train_min = get_max_min(training_X)
    training_X = min_max_feature_scaling(training_X, train_max, train_min)

    print("Test on Training set:", train, "with k =", k, "and distance_metric =", distance_metric)
    training_n = knn(training_X, training_y, training_X, training_y, k, distance_metric)
    training_acc = training_n / len(training_y)
    print("Training Prediction Accuracy:", training_n, "/", len(training_y), "=", training_acc, ",",
          round(training_acc * 100, 2), "%\n")

    test_X, test_y = read_data(test)
    test_X = min_max_feature_scaling(test_X, train_max, train_min)

    print("Test on Test set:", test, "with k =", k, "and distance_metric =", distance_metric)
    test_n = knn(test_X, test_y, training_X, training_y, k, distance_metric)
    test_acc = test_n / len(test_y)
    print("Testing Prediction Accuracy:", test_n, "/", len(test_y), "=", test_acc, ",",
          round(test_acc * 100, 2), "%\n")

    print("Training vs. Test difference:", round(training_acc * 100, 2), "% -", round(test_acc * 100, 2), "% =",
          round((training_acc - test_acc) * 100, 2), "%")
