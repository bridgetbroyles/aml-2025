''' notes:
Python allows negative indexing in lists: list[-1] = last element.

Perceptron
A simple linear classifier that finds a line or hyperplane to separate two classes. It computes a weighted sum of inputs plus a bias, then outputs 1 if the sum ≥ 0 or 0 otherwise.

Bias
An extra weight added to the sum (not tied to any feature). It shifts the decision boundary away from the origin, letting the model fit data that aren’t centered at zero.

train_data is a list of rows. Each row is [feature1, feature2, …, featureN, label]. Features are inputs, label (row[-1]) is the true class (0 or 1). Weights is a list of N+1 numbers: one per feature plus the bias.
'''


from random import uniform
from random import randrange

#-------------------------#

def classify(row, weights):
 activation = weights[-1]                               # might be unneeded
   for i in range(len(row)-1):                            # might be not -1
        activation += weights[i] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

#-------------------------#

def train(train_data, n_epochs, l_rate=1):
    weights = []
    for i in range(len(train_data[0])):  # one weight for each feature (including bias)
        random_weight = uniform(-1, 1)
        weights.append(random_weight)

    # Train for the given number of epochs
    for epoch in range(n_epochs):
        num_correct = 0 

        for row in train_data:
            # Get the prediction from the current weights
            prediction = classify(row, weights)

            # Calculate the error (actual - predicted)
            actual = row[-1]
            error = actual - prediction

            # If there's no error, the prediction is correct
            if error == 0:
                num_correct += 1
            else:
                # Update the bias weight (last weight)
                weights[-1] = weights[-1] + l_rate * error

                # Update the rest of the weights
                for i in range(len(row) - 1):                  # exclude the label at the end
                    weights[i] = weights[i] + l_rate * error * row[i]

        # Print percentage of correctly classified records for this epoch
        accuracy = num_correct / len(train_data)
        print("epoch", epoch, "....", round(accuracy * 100), "% correct.")

    return weights

#-------------------------#

# TASK 4 LATER IN DAY #, unedited by me

# 1. Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    # Count how many predictions match the true labels
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    # Return percent correct
    return (correct / len(actual)) * 100

# 2. Split a dataset into k roughly equal folds
def cross_validation_split(dataset, n_folds):
    dataset_copy = list(dataset)           # make a shallow copy
    fold_size = len(dataset) // n_folds    # integer size of each fold
    folds = []                             # list to hold all folds

    for _ in range(n_folds):
        fold = []
        # Keep picking random rows until fold reaches desired size
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            row = dataset_copy.pop(index)
            fold.append(row)
        folds.append(fold)
    return folds

# 3. Run cross‑validation using your train() and classify() functions
def cross_validate(dataset, n_folds, n_epochs, l_rate=1):
    folds = cross_validation_split(dataset, n_folds)
    scores = []

    # For each fold, use it as the test set and the rest as training set
    for fold in folds:
        # Build training set by concatenating all other folds
        train_set = []
        for other in folds:
            if other is not fold:
                train_set.extend(other)

        # Prepare test set: copy each row but blank out its label
        test_set = []
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)

        # Train the perceptron on train_set
        weights = train(train_set, n_epochs, l_rate)

        # Make predictions on the test_set
        predicted = []
        for row in test_set:
            yhat = classify(row, weights)
            predicted.append(yhat)

        # Extract the true labels from the original fold
        actual = [row[-1] for row in fold]

        # Compute accuracy for this fold
        acc = accuracy_metric(actual, predicted)
        scores.append(acc)

    return scores
