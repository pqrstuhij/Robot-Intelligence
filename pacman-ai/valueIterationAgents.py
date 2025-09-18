# valueIterationAgents.py

from learningAgents import ValueEstimationAgent
import mdp, util

class ValueIterationAgent(ValueEstimationAgent):
    """
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
          
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  

        
        for i in range(iterations):
            newValues = util.Counter()
            for state in mdp.getStates():
                if mdp.isTerminal(state):
                    continue
                actionValues = []
                for action in mdp.getPossibleActions(state):
                    actionValues.append(self.computeQValueFromValues(state, action))
                if actionValues:
                    newValues[state] = max(actionValues)
            self.values = newValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        q_value = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            q_value += prob * (reward + self.discount * self.values[nextState])
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit. Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None

        best_action = None
        best_value = float('-inf')
        for action in self.mdp.getPossibleActions(state):
            q_value = self.computeQValueFromValues(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action

    def getPolicy(self, state):
        """
          Returns the policy at the state (no exploration).
        """
        return self.computeActionFromValues(state)

    def getAction(self, state):
        """
          Returns the action to take in the current state.
          Note that because this is a value iteration agent,
          the action returned is simply the best action
          according to the value function stored in self.values.
        """
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        """
          Returns the Q-value of the (state, action) pair.
          The Q-value is computed using the value function
          stored in self.values.
        """
        return self.computeQValueFromValues(state, action)

