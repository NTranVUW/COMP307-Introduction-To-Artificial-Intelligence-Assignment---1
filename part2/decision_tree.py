import sys
import numpy as np

baseline_instances = None


# Class for a leaf node - the final node in the branch
class leaf:
    def __init__(self, data, probability):
        self.data = data
        # get the most common class
        self.prediction = np.unique(data, return_counts=True)[0][np.argmax(np.unique(data, return_counts=True)[1])]
        self.probability = probability

    def report(self, indent):
        if self.probability == 0:
            print("{}Unknown".format(indent))
        else:
            print("{}Class {}, prob={}".format(indent, self.prediction, self.probability))


# Class for a decision node - the node that splits the data
class Node:
    def __init__(self, attribute, left, right):
        self.attribute = attribute
        self.left = left
        self.right = right

    def report(self, indent):
        print("{}{} = True:".format(indent, self.attribute))
        self.left.report(indent + "\t")
        print("{}{} = False:".format(indent, self.attribute))
        self.right.report(indent + "\t")


# Compare the instances to the classes predicted by the tree
def accuracy(instances, predictions):
    correct = 0
    for i in range(len(instances)):
        # print("Actual {}, Predictions {}".format(instances[i][0], predictions[i]))
        if instances[i][0] == predictions[i]:
            correct += 1
    accuracy = correct / len(instances)
    print("Accuracy: {}/{} = {}%".format(correct, len(instances), round(accuracy, 2) * 100))
    return accuracy


# Traverse the tree and make predictions
def predict(instance, attributes, root):
    if isinstance(root, leaf):
        return root.prediction
    else:
        if instance[np.where(attributes == root.attribute)] == 'true':
            return predict(instance, attributes, root.left)
        else:
            return predict(instance, attributes, root.right)


# weighted impurity = left weight * left impurity + right weight * right impurity
def weighted_impurity(instances_left, instances_right):
    n = len(instances_left) + len(instances_right)
    weight_left = len(instances_left) / n
    weight_right = len(instances_right) / n

    # convert to numpy array for np.unique function
    instances_left = np.array(instances_left)
    instances_right = np.array(instances_right)
    unique_counts_left = np.unique(instances_left[:, 0], return_counts=True)
    unique_counts_right = np.unique(instances_right[:, 0], return_counts=True)

    probability_left = []
    probability_right = []

    for count in unique_counts_left[1]:
        probability_left.append(count / len(instances_left))

    for count in unique_counts_right[1]:
        probability_right.append(count / len(instances_right))

    purity_left = np.prod(probability_left)
    purity_right = np.prod(probability_right)

    weighted_impurity = weight_left * purity_left + weight_right * purity_right
    return weighted_impurity


# Split the data into two groups based on the attribute - one subset for True and one for False
def split_instances(instances, i):
    instances_left = []
    instances_right = []
    for instance in instances:
        if instance[i] == 'true':
            instances_left.append(instance)
        else:
            instances_right.append(instance)
    return np.array(instances_left), np.array(instances_right)


# Build the decision tree - returns the root node
def build_tree(instances, attributes):
    # if instances is empty, return a leaf node with the most common class
    if len(instances) == 0:
        unique_counts = np.unique(baseline_instances[:, 0], return_counts=True)
        # Baseline classifier - get the most common class
        probability = unique_counts[1][np.argmax(unique_counts[1])] / len(baseline_instances)
        return leaf(baseline_instances, probability)
    # if all the instances have the same class, return a leaf node with that class and probability 1
    elif np.unique(instances[:, 0]).size == 1:
        return leaf(instances[:, 0], 1.0)
    # if attributes is empty, return a leaf node with the most common class and probability of most common class
    elif len(attributes) == 0:
        probability = np.unique(instances[:, 0], return_counts=True)[1][
                          np.argmax(np.unique(instances[:, 0], return_counts=True)[1])] / len(instances)
        return leaf(instances[:, 0], probability)
    else:
        best_attribute = 0
        best_impurity = 1
        best_instances_left = None
        best_instances_right = None
        for i, attribute in enumerate(attributes, start=1):
            instances_left, instances_right = split_instances(instances, i - 1)
            if len(instances_left) == 0 or len(instances_right) == 0:
                impurity = 1
            else:
                impurity = weighted_impurity(instances_left[:, [0, i - 1]], instances_right[:, [0, i - 1]])

            if impurity <= best_impurity:
                best_impurity = impurity
                best_attribute = i - 1
                best_instances_left = instances_left
                best_instances_right = instances_right

        # remove the attribute with the lowest impurity from the list of attributes
        remaining_attributes = np.delete(attributes, best_attribute)
        remaining_instances_left = []
        remaining_instances_right = []

        # split the remaining instances into two groups based on the attribute and delete the attribute from the
        # instances
        if len(best_instances_left) != 0:
            remaining_instances_left = np.delete(best_instances_left, best_attribute, 1)

        if len(best_instances_right) != 0:
            remaining_instances_right = np.delete(best_instances_right, best_attribute, 1)

        # go down the tree recursively
        left_tree = build_tree(remaining_instances_left, remaining_attributes)
        right_tree = build_tree(remaining_instances_right, remaining_attributes)

        # return the root node
        return Node(attributes[best_attribute], left_tree, right_tree)


# Used for cross validation
def cross_validate(training, testing, k):
    training_accuracies = []
    testing_accuracies = []
    print("\n{}-fold cross validation".format(k))
    for i in range(k):
        print("\nFold: ", i)
        train = training + "-run-" + str(i)
        test = testing + "-run-" + str(i)

        train_instances, train_attributes = read_data(train)
        baseline_instances = train_instances

        print("Baseline...")
        unique_counts = np.unique(baseline_instances[:, 0], return_counts=True)
        baseline_accuracy = unique_counts[1][1] / len(baseline_instances)
        print("Accuracy: {}/{} = {}%".format(unique_counts[1][1], len(baseline_instances),
                                             round(baseline_accuracy * 100, 2)))

        tree = build_tree(train_instances, train_attributes)
        print("Training...")
        training_predictions = []
        for instance in train_instances:
            prediction = predict(instance, train_attributes, tree)
            training_predictions.append(prediction)
        training_accuracy = accuracy(train_instances, training_predictions)
        training_accuracies.append(training_accuracy)

        print("Testing...")
        testing_instances, testing_attributes = read_data(test)
        testing_predictions = []
        for instance in testing_instances:
            prediction = predict(instance, test_attributes, tree)
            testing_predictions.append(prediction)
        testing_accuracy = accuracy(testing_instances, testing_predictions)
        testing_accuracies.append(testing_accuracy)

    print("\nAverage training accuracy: ", round(np.mean(training_accuracies), 2) * 100, "%")
    print("Average testing accuracy: ", round(np.mean(testing_accuracies), 2) * 100, "%")


# Splits the data into instances and attributes
def read_data(file):
    with open(file, 'r') as f:
        data = f.readlines()

    data = [line.strip().split() for line in data]
    data = np.array(data)
    instances = data[1:]
    attributes = data[0, 1:]
    return instances, attributes


if __name__ == '__main__':
    # test = 'hepatitis-test'
    # training = 'hepatitis-training'

    # test = 'golf-test'
    # training = 'golf-training'

    if len(sys.argv) == 3:
        test = sys.argv[2]
        training = sys.argv[1]
    else:
        raise Exception("Usage: decision_tree.py <training-file> <test-file>")

    instances, attributes = read_data(training)
    baseline_instances = instances

    print("Building tree...")
    tree = build_tree(instances, attributes)
    print("Printing tree...")
    tree.report("")

    print("Evaluating tree...")

    print("...Baseline accuracy...")
    unique_counts = np.unique(baseline_instances[:, 0], return_counts=True)
    baseline_accuracy = unique_counts[1][1] / len(baseline_instances)
    print("{}/{} = {}%".format(unique_counts[1][1], len(baseline_instances), round(baseline_accuracy * 100, 2)))

    print("...On training data...")
    training_predictions = []
    for instance in instances:
        prediction = predict(instance, attributes, tree)
        training_predictions.append(prediction)
    accuracy(instances, training_predictions)

    test_instances, test_attributes = read_data(test)
    print("...On test data...")
    test_predictions = []
    for instance in test_instances:
        prediction = predict(instance, test_attributes, tree)
        test_predictions.append(prediction)
    accuracy(test_instances, test_predictions)

    cross_validate(training, test, 10)
