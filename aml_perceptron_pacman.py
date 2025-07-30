'''
notes
'''

import Helpers.util
from Exercises.perceptron_multiclass import PerceptronClassifier
from random import randrange
from pacman import GameState

class PerceptronClassifierPacman(PerceptronClassifier):
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

    #### MINE
    def classify(self, data):

        features, legal_moves = self.convert_data(data)
        
        #Compute the output value for each possible label
        outputs = []
        for l in legal_moves:
            feat = features[l] #grab the features for this one action
            dotProduct = 0
            for i in range(len(feat)):
                #print("-- mine", self.weights[i], feat[i])
                dotProduct += self.weights[i] * feat[i]
            #print("mine", dotProduct)
            outputs.append(dotProduct)

        #print("mine", legal_moves, outputs)
        #Return the label with the highest output value
        maxScore = outputs[0]
        maxLabel = legal_moves[0]
        for i in range(len(outputs)):
            if outputs[i] > maxScore:
                maxScore = outputs[i]
                maxLabel = legal_moves[i]
        return [maxLabel] #needs to be returned in a list
        #### END MINE ###
 
    

    def train(self, trainingData, trainingLabels):

        #init weights
        self.weights = [randrange(-1,1) for x in range(14)] #[[randrange(-1,1) for x in range(len(trainingData[0]))] for y in range(len(self.legalLabels))]
        
        for iteration in range(self.epochs):
            num_correct = 0
            for i in range(len(trainingData)):

                real = trainingLabels[i]
                pred = self.classify(trainingData[i])[0] #returned as a list, pull it out of the list
            
                #convert the data record
                features, legal_moves = self.convert_data(trainingData[i])
                #print(len(features[legal_moves[0]]))
                
                features_real = features[real] #grab the features for the actual            # features[real] returns the list paired with the choice
                features_pred = features[pred] #grab the features for the pred               # ex {'East', {.2 , 4 , 5, 6 , ... , .3 }} 
                                                                                            # the numbers are basically how good of a move East is, data for weights

                if real != pred:
                    for j in range(len(features_real)):
                        self.weights[j] += features_real[j] 
                        self.weights[j] -= features_pred[j]   
                else:
                    num_correct += 1

            print("epoch", iteration, "....", round(num_correct/len(trainingData), 2) * 100, "% correct.")








