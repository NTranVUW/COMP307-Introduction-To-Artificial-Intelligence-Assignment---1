import sys
import numpy as np


# Used to test the perceptron algorithm
def predict(instances, weights, bias):
    predictions = np.zeros(len(instances))
    for i, instance in enumerate(instances):
        weighted_sum = sum(instance * weights) + bias
        predicted_class = activation_function(weighted_sum)
        predictions[i] = predicted_class
    return predictions


# Returns the accuracy of the perceptron's predictions
def prediction_accuracy(predictions, classes):
    correct = 0
    for i, class_ in enumerate(classes):
        if predictions[i] == class_:
            correct += 1
    return correct / len(classes)


# Update weights: w = w + eta(d - y)x
def update_weights(weights, desired, output, features, learning_rate):
    return weights + learning_rate * (desired - output) * features


# Activation function, returns 1 if input > 0 or 0 if input < 0
def activation_function(value):
    return 1 if value > 0 else 0


# Perceptron algorithm:
# 1. Initialize weights to random values
# 2. For each instance:
#   a. Calculate the weighted sum
#   b. Calculate the predicted class
#   c. Update the weights if the predicted class is incorrect
# 3. Repeat until all instances are correctly classified or the maximum number of epochs is reached
def perceptron(instances, classes, learning_rate):
    # Initialise weights randomly
    np.random.seed(123)
    weights = np.random.rand(len(instances[0]))

    # Initialise the bias randomly
    bias = np.random.rand()

    epochs = 100  # Number of epochs (iterations without improvement)
    current_predictions = np.zeros(len(classes))  # Initialise predictions to 0

    # The best weights and bias found so far
    max_accuracy = 0
    max_weights = np.zeros(len(instances[0]))
    max_bias = 0

    max_iter = 0
    # If accuracy does not improve for 100 epochs, stop training
    while max_iter < epochs:
        acc = prediction_accuracy(current_predictions.astype(int), classes)

        # If the accuracy is better than the best accuracy so far, update the best accuracy and weights
        if acc > max_accuracy:
            max_accuracy = acc
            max_weights = weights
            max_bias = bias
            max_iter = 0
        else:
            max_iter += 1

        # If accuracy is 100% then return the best weights and bias
        if acc == 1:
            return max_weights, max_bias

        # Calculate the output and classify the instances
        for i, instance in enumerate(instances):
            weighted_sum = sum(instance * weights) + bias
            predicted_class = activation_function(weighted_sum)
            current_predictions[i] = predicted_class  # Store the prediction

            # Update weights and bias if the prediction is incorrect
            if predicted_class != classes[i]:
                weights = update_weights(weights, classes[i], predicted_class, instance, learning_rate)
                bias = bias + learning_rate * (classes[i] - predicted_class) * 1

    return max_weights, max_bias, current_predictions


# Encode labels to 0 and 1
def label_encoder(y):
    unique_labels = np.unique(y)
    nums = np.array(range(len(unique_labels)))
    encoder = dict(zip(unique_labels, nums))
    y = np.array([encoder[label] for label in y])
    return y


def read_data(file):
    with open(file, 'r') as f:
        data = f.readlines()

    data = [line.split() for line in data]
    data = np.array(data[1:])
    np.random.shuffle(data)  # Randomise the data so that the final accuracy is not dependent on the order of the data
    X = data[:, :-1].astype(float)
    y = label_encoder(data[:, -1])
    return X, y


# Run the perceptron algorithm on the training data and test on the test data
def train_and_test(X, y, learning_rate, train_size):
    # Split the data into training and test sets
    training_X = X[:int(len(X) * train_size)]
    training_y = y[:int(len(y) * train_size)]
    testing_X = X[int(len(X) * train_size):]
    testing_y = y[int(len(y) * train_size):]

    print("Training perceptron with learning rate {}".format(learning_rate))
    trained_weights, trained_bias, predictions = perceptron(training_X, training_y, learning_rate)

    print("\nTraining Accuracy: {}%".format(round(prediction_accuracy(predictions, training_y) * 100, 2)))
    print("\nTrained weights: {}".format(trained_weights))
    print("\nTrained bias: {}".format(trained_bias))

    print("\nTesting perceptron with learning rate {}".format(learning_rate))
    testing_predictions = predict(testing_X, trained_weights, trained_bias)
    print("\nTest Accuracy: {}%".format(round(prediction_accuracy(testing_predictions, testing_y) * 100, 2)))


# Run the perceptron algorithm on the training data
def training_only(X, y, learning_rate):
    print("Training perceptron with learning rate {}\n".format(learning_rate))
    trained_weights, trained_bias, predictions = perceptron(X, y, learning_rate)

    print("\nAccuracy: {}%".format(round(prediction_accuracy(predictions, y) * 100, 2)))
    print("\nTrained weights: {}".format(trained_weights))
    print("\nTrained bias: {}".format(trained_bias))


if __name__ == '__main__':
    # Hardcoded data for testing
    # X, y = read_data("ionosphere.data")
    # learning_rate = 0.2
    # train_size = 0.8
    # train_and_test(X, y, learning_rate, train_size)

    if len(sys.argv) > 2:
        X, y = read_data(sys.argv[1])
        learning_rate = float(sys.argv[2])
    else:
        raise Exception("Usage: perceptron.py <data_file> <learning_rate> <train_size>(optional)")

    # Both train and test the perceptron
    if len(sys.argv) == 4:
        train_size = float(sys.argv[3])
        if train_size > 1 or train_size < 0:
            raise Exception("Train size must be between 0 and 1")
        train_and_test(X, y, learning_rate, train_size)

    # Only train the perceptron
    if len(sys.argv) == 3:
        training_only(X, y, learning_rate)
