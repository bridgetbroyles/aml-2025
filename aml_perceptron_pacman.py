'''
notes
'''

import Helpers.util
from Exercises.perceptron_multiclass import PerceptronClassifier
from random import randrange
from pacman import GameState

class PerceptronClassifierPacman(PerceptronClassifier):
  #prewritten------------------------------------------------------------------#
    def __init__(self, legalLabels, maxIterations):
        PerceptronClassifier.__init__(self, legalLabels, maxIterations)
        self.weights = [] #Helpers.util.Counter()
        self.epochs = maxIterations
    

    def convert_data(self, data):
        # fix datatype issues
        # if it comes in inside of a list, pull it out of the list
        if isinstance(data, list):
            data = data[0]

        #data comes as a tuple
        all_moves_features = data[0] #grab the features (dict of action->features)
        legal_moves = data[1]        #grab the list of legal moves from this state
        return_features = {}
        #loop each action
        for key in all_moves_features:
            #convert feature values from dict to list
            all_features = ['foodCount', 'STOP', 'nearest_ghost', 'ghost-0', 'capsule-0', 'food-0', 'food-1', \
                            'food-2', 'food-3', 'food-4', 'capsule count', 'win', 'lose', 'score']
            dict_features = all_moves_features[key] 
            list_features = []
            # grab all feature values & put them in a list
            for feat in all_features:
                if feat not in dict_features:
                    list_features.append(0)
                else:
                    list_features.append(dict_features[feat])

            print(len(list_features))
            return_features[key] = list_features

        return (return_features, legal_moves) 

 #endl prewritten------------------------------------------------------------------#
  
    def classify(self, data):
    # Convert the input into features and legal moves
    features, legal_moves = self.convert_data(data)

    # Compute scores for each legal move
    outputs = []
    for move in legal_moves:
        move_features = features[move]
        score = 0
        for i in range(len(move_features)):
            score += self.weights[i] * move_features[i]
        outputs.append(score)

    # Choose the move with the highest score
    max_score = outputs[0]
    best_move = legal_moves[0]
    for i in range(len(outputs)):
        if outputs[i] > max_score:
            max_score = outputs[i]
            best_move = legal_moves[i]

    return [best_move]
#-------------------------#
def train(self, trainingData, trainingLabels):
    # Initialize weights (14 features)
    self.weights = []
  
    for i in range(14):
        r = randrange(-1, 1)
        self.weights.append(r)

    for iteration in range(self.epochs):
        correct = 0
        for i in range(len(trainingData)):
            actual = trainingLabels[i]
            predicted = self.classify(trainingData[i])[0]

            # Get feature vectors for both predicted and actual moves
            features, legal_moves = self.convert_data(trainingData[i])
            actual_feat = features[actual]
            predicted_feat = features[predicted]

            if actual != predicted:
                for j in range(len(actual_feat)):
                    self.weights[j] += actual_feat[j]
                    self.weights[j] -= predicted_feat[j]
            else:
                correct += 1

        print("epoch", iteration, "....", round(correct / len(trainingData), 2) * 100, "% correct.")
#-------------------------#

          
