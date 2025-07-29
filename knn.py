'''
notes
unstudied or read
'''


from math import sqrt
from collections import Counter
from csv import reader

# This function makes all the data between 0 and 1 so it's easier to compare
def normalize_data(train, test):
    min_vals = []
    max_vals = []
    
    # Find smallest and biggest number in each column of the training set
    for col in zip(*train):
        min_vals.append(min(col))
        max_vals.append(max(col))

    # This part changes all numbers to be between 0 and 1
    def normalize(dataset):
        normalized_data = []
        for row in dataset:
            new_row = []
            for i in range(len(row)):
                min_val = min_vals[i]
                max_val = max_vals[i]
                if max_val - min_val != 0:
                    norm_val = (row[i] - min_val) / (max_val - min_val)
                else:
                    norm_val = 0
                new_row.append(norm_val)
            normalized_data.append(new_row)
        return normalized_data

    return normalize(train), normalize(test)

# This function finds how far two points are using distance formula
def get_distances(point, data):
    distances = []
    for i in range(len(data)):
        total = 0
        for j in range(len(point)):
            total += (point[j] - data[i][j]) ** 2
        dist = sqrt(total)
        distances.append(dist)
    return distances

# This is the k-nearest neighbors algorithm
def run_knn(train_set, test_set, k):
    # Split inputs and labels for both train and test sets
    X_train = []
    y_train = []
    for row in train_set:
        X_train.append(row[:-1])
        y_train.append(row[-1])

    X_test = []
    y_test = []
    for row in test_set:
        X_test.append(row[:-1])
        y_test.append(row[-1])

    y_pred = []

    for i in range(len(X_test)):
        test_point = X_test[i]
        distances = get_distances(test_point, X_train)

        # Sort distances manually by index
        sorted_indices = list(range(len(distances)))
        for i in range(len(sorted_indices)):
            for j in range(i + 1, len(sorted_indices)):
                if distances[sorted_indices[j]] < distances[sorted_indices[i]]:
                    sorted_indices[i], sorted_indices[j] = sorted_indices[j], sorted_indices[i]

        # Get the labels of the k closest points
        nearest_neighbors = []
        for i in range(k):
            nearest_neighbors.append(y_train[sorted_indices[i]])

        # Count which label appears most
        count = {}
        for label in nearest_neighbors:
            if label not in count:
                count[label] = 1
            else:
                count[label] += 1

        most_common = None
        most_count = 0
        for label in count:
            if count[label] > most_count:
                most_common = label
                most_count = count[label]

        y_pred.append(most_common)

    return y_pred, y_test

# This tests how good the algorithm is using cross-validation
def run_CV(data, k=3, folds=5):
    fold_size = len(data) // folds
    accuracy = 0.0

    for i in range(folds):
        train_set = []
        test_set = []

        # Make test set
        for j in range(len(data)):
            if j >= i * fold_size and j < (i + 1) * fold_size:
                test_set.append(data[j])
            else:
                train_set.append(data[j])

        # Normalize the numbers
        normalized_train, normalized_test = normalize_data(train_set, test_set)

        # Run k-NN and check how many were correct
        y_pred, y_test = run_knn(normalized_train, normalized_test, k)
        correct = 0
        for i in range(len(y_test)):
            if y_pred[i] == y_test[i]:
                correct += 1

        fold_accuracy = correct / len(y_test)
        accuracy += fold_accuracy

    # Return average accuracy across all folds
    return accuracy / folds
