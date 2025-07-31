''' notes
changed
'''

import Helpers.util
from random import randrange

class PerceptronClassifier:
  
# already written
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.epochs = max_iterations
        self.weights = None 
# endl pre-written

  def classify(self, data):
    # Find score for each label by doing dot product with weights
    scores = []
    for label_index in range(len(self.legalLabels)):    # legal labels is just 0 - 9
        score = 0
        for i in range(len(data)):
            score += self.weights[label_index][i] * data[i]   # data is flattened 2D array of the values in each pixel
        scores.append(score)

    # Find the label with the biggest score
    best_score = scores[0]
    best_label = 0
    for i in range(len(scores)):
        if scores[i] > best_score:
            best_score = scores[i]
            best_label = i
    return best_label # return label not best score

  def train( self, trainingData, trainingLabels):
          self.weights = [[randrange(-1,1) for x in range(len(trainingData[0]))] for y in range(len(self.legalLabels))]
          for iteration in range(self.epochs):
              num_correct = 0
              for i in range(len(trainingData)):
                  real = trainingLabels[i]
                  pred = self.classify(trainingData[i])
                  if real != pred:
                      for j in range(len(trainingData[i])):
                        # lower score of wrong answer, raise score of correct answer, 
                        # we lower because since classify returns the maxLabel, the maxLabel that 
                        # this wrong answer has is too large.
                          self.weights[real][j] += trainingData[i][j]
                          self.weights[pred][j] -= trainingData[i][j]
                  else:
                      num_correct += 1
              print("epoch", iteration, "....", round(num_correct/len(trainingData), 2) * 100, "% correct.")
  

