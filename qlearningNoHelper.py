## qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.

from Helpers.game import *
from Helpers.learningAgents import ReinforcementAgent
from Helpers.featureExtractors import *

import random, Helpers.util, math

class QLearningAgent(ReinforcementAgent):
    """
    A Q-learning agent learns how to choose actions
    that lead to the most reward over time.
    """
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.qVals = {}  # Stores Q-values for state-action pairs

    def getQValue(self, state, action):
        action = action.capitalize()
        if state not in self.qVals:
            # Initialize all possible actions to 0
            self.qVals[state] = {'North': 0, 'South': 0, 'East': 0, 'West': 0, 'Stop': 0, 'Exit': 0}
        return self.qVals[state][action]

    def computeValueFromQValues(self, state):
        """
        Return the maximum Q-value over all legal actions.
        If there are no legal actions, return 0.0
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        # Compute Q-values and return the maximum
        values = [self.getQValue(state, action) for action in legalActions]
        return max(values)

    def computeActionFromQValues(self, state):
        """
        Compute the best action to take in this state.
        If there is a tie, choose randomly among best.
        If there are no legal actions, return None.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        # Find the highest Q-value
        bestValue = float('-inf')
        bestActions = []
        for action in legalActions:
            q = self.getQValue(state, action)
            if q > bestValue:
                bestValue = q
                bestActions = [action]
            elif q == bestValue:
                bestActions.append(action)
        return random.choice(bestActions)

    def getAction(self, state):
        """
        Choose an action: with probability epsilon choose randomly,
        otherwise choose best policy action.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        if random.random() < self.epsilon:
            return random.choice(legalActions)
        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        action = action.capitalize()
        # Classic Q-learning update
        oldQ = self.getQValue(state, action)
        futureReward = self.computeValueFromQValues(nextState)
        sample = reward + self.discount * futureReward
        self.qVals[state][action] = oldQ + self.alpha * (sample - oldQ)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
