# qlearningAgents.py
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
    This class is for a Q-Learning Agent.
    
    A Q-learning agent learns how to choose actions
    that lead to the most reward over time.
    
    Functions you will look at:
      - getQValue
      - computeValueFromQValues
      - computeActionFromQValues
      - getAction
      - update

    You can use these variables:
      - self.epsilon: how often to choose a random action (exploration)
      - self.alpha: how fast to learn (learning rate)
      - self.discount: how much future rewards matter (discount factor)
    
    You can use this function:
      - self.getLegalActions(state): gives you a list of valid actions in a state
    """
    
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.qVals = {}  # Stores Q-values for state-action pairs

    def getQValue(self, state, action):
        action = action.capitalize()  # Make sure the action is capitalized
        """
        This function gives back the Q-value for a specific state and action.
        If this state has never been seen before, create default values.
        """
        if state not in self.qVals:
            # If we've never seen this state, add it with all actions set to 0
            self.qVals[state] = {'North':0, 'South':0, 'East':0, 'West':0, 'Stop':0, 'Exit':0}
        
        return self.qVals[state][action]

    def getActionAndValue(self, state):
        """
        This is a helper function.
        It returns the best action and its Q-value in the given state.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return (None, 0.0)

        bestValue = float('-inf')
        bestActions = []

        for action in legalActions:
            value = self.getQValue(state, action)
            if value > bestValue:
                bestValue = value
                bestActions = [action]  # Found a new best action
            elif value == bestValue:
                bestActions.append(action)  # Tie, so add to list

        # Pick one of the best actions randomly if there's a tie
        return (random.choice(bestActions), bestValue)

    def computeValueFromQValues(self, state):
        """
        This function returns the best Q-value you can get from this state.
        It looks at all legal actions and picks the highest Q-value.
        """
        return self.getActionAndValue(state)[1]

    def computeActionFromQValues(self, state):
        """
        This function returns the best action to take in this state.
        If there's a tie, it picks randomly from the best actions.
        If there are no legal actions, it returns None.
        """
        return self.getActionAndValue(state)[0]

    def getAction(self, state):
        """
        This function decides which action to take in the current state.
        Sometimes it will choose a random action (exploration),
        and sometimes it will pick the best action (exploitation).

        The chance of choosing a random action is controlled by self.epsilon.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        if random.random() < self.epsilon:
            # Explore: pick a random action
            return random.choice(legalActions)
        else:
            # Exploit: pick the best action
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        action = action.capitalize()
        """
        This function updates the Q-value for the given state and action.
        It's called every time we take an action and see the result.

        Here's how it works:
        - old Q-value is what we already know
        - new Q-value = reward + discounted future reward
        - we move our Q-value a little toward the new value
        """
        oldQ = self.getQValue(state, action)
        futureReward = self.computeValueFromQValues(nextState)
        newQ = reward + self.discount * futureReward
        self.qVals[state][action] = oldQ + self.alpha * (newQ - oldQ)

    # These two functions are just easier names for computing the best action or value
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
