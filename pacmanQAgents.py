from Exercises.qlearningAgents import QLearningAgent
import Helpers.util
from Helpers.featureExtractors import *

class PacmanQAgent(QLearningAgent):
    "Same as QLearningAgent but with different default settings"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        # Set default learning parameters (can be changed via command line)
        args['epsilon'] = epsilon      # chance to explore randomly
        args['gamma'] = gamma          # discount rate for future rewards
        args['alpha'] = alpha          # learning rate
        args['numTraining'] = numTraining  # how many episodes to train for
        self.index = 0  # Pacman is always agent index 0
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        # Use the parent's getAction method to decide what to do
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)  # tell the game what action Pacman took
        return action

class ApproximateQAgent(PacmanQAgent):
    """
    Q-learning agent that uses feature-based approximation instead of a table.
    Only getQValue and update need to be changed.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        # Initialize feature extractor and weights dictionary
        self.featExtractor = Helpers.util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = Helpers.util.Counter()  # dictionary for feature weights

    def getWeights(self):
        # Return the current weights dictionary
        return self.weights

#--------------#

    def getQValue(self, state, action):
        # Calculate Q(state,action) = sum of (weight * feature value)
        feats = self.featExtractor.getFeatures(state, action)
        qVal = 0.0
        for feat in feats:
            qVal += self.weights[feat] * feats[feat]  # multiply weight by feature value
        return qVal

    def update(self, state, action, nextState, reward):
        # Update weights based on how much better or worse the outcome was
        feats = self.featExtractor.getFeatures(state, action)
        oldQVal = self.getQValue(state, action)
        target = reward + self.discount * self.computeValueFromQValues(nextState)
        difference = target - oldQVal
        for feat in feats:
            self.weights[feat] += self.alpha * difference * feats[feat]  # update rule
#------#
    def final(self, state):
        # Called at the end of each game
        #-
        PacmanQAgent.final(self, state)

        # Check if training is done
        if self.episodesSoFar == self.numTraining:
            # You could print weights here to debug if you want
            pass
