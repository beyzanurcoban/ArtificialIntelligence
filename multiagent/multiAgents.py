# multiAgents.py
# BEYZANUR COBAN 64763
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from typing import Mapping
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0
        minFoodDist = 999999
        minGhostDist = 999999

        stopPenalty = -10
        ghostPenalty = -50
        losePenalty = -5000
        foodReward = 20

        foodList = newFood.asList()

        if (len(foodList) == 1):
            foodReward = 80
    
        for food in foodList:
            minFoodDist = min(minFoodDist, manhattanDistance(newPos, food))
        
        for ghostState in newGhostStates:
            minGhostDist = min(minGhostDist, manhattanDistance(newPos, ghostState.getPosition()))
            if (minGhostDist == 0):
                return losePenalty

        score = (foodReward/(minFoodDist)) + (ghostPenalty/minGhostDist)

        if (action == Directions.STOP):
            score = score + stopPenalty

        return successorGameState.getScore() + score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        PACMAN = 0
        return self.getBestAction(gameState, PACMAN, 0)


    # Pacman tries to maximize the score
    def maxVal(self, gameState, agent, depth):
        value = float('-inf')

        legalActions = gameState.getLegalActions(agent)

        for action in legalActions:
            successor = gameState.generateSuccessor(agent,action)

            # Since only PACMAN calls this function, successorAgent will always be 1
            # But in the case of someone calls this function with the wrong agent, I added this part
            successorAgent = agent + 1

            if (successorAgent == 1):
                value = max(value, self.value(successor, successorAgent, depth))
            else:
                value = float('-inf')
        
        return value

    # Ghosts try to minimize the score
    def minVal(self, gameState, agent, depth):
        value = float('inf')
        numOfAgents = gameState.getNumAgents()

        legalActions = gameState.getLegalActions(agent)

        for action in legalActions:
            successor = gameState.generateSuccessor(agent,action)

            successorAgent = agent + 1
            # If the successor agent is PACMAN, then the agent index will 0 in the function call
            # Also, this means that PACMAN and the ghosts finished their moves, so we should increase the depth
            if (successorAgent == numOfAgents):
                value = min(value, self.value(successor, 0, depth+1))
            else:
                value = min(value, self.value(successor, successorAgent, depth))

        return value


    def value(self, gameState, agent, depth):
        # If we reach the leaf node
        if (gameState.isWin() or gameState.isLose() or self.depth == depth):
            return self.evaluationFunction(gameState)

        # If the agent is Pacman, calculate MAX. Else, calculate MIN.
        if (agent == 0):
            return self.maxVal(gameState, agent, depth)
        else:
            return self.minVal(gameState, agent, depth)


    def getBestAction(self, gameState, agent, depth):
        value = float('-inf')
        legalActions = gameState.getLegalActions(0)

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)

            succValue = self.value(successor, agent+1, depth)

            if (succValue > value):
                value = succValue
                act = action

        return act


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        PACMAN = 0
        DEPTH = 0
        return self.getBestAction(gameState, PACMAN, DEPTH)


    def maxVal(self, gameState, agent, depth, alpha, beta):
        value = float('-inf')

        legalActions = gameState.getLegalActions(agent)

        for action in legalActions:
            successor = gameState.generateSuccessor(agent,action)
            successorAgent = agent + 1

            value = max(value, self.value(successor, successorAgent, depth, alpha, beta))

            if (value > beta):
                return value

            alpha = max(alpha, value)
        
        return value
  
    def minVal(self, gameState, agent, depth, alpha, beta):
        value = float('inf')
        numOfAgents = gameState.getNumAgents()

        legalActions = gameState.getLegalActions(agent)

        for action in legalActions:
            successor = gameState.generateSuccessor(agent,action)

            successorAgent = agent + 1
            # If the successor agent is PACMAN, then the agent index will 0 in the function call
            # Also, this means that PACMAN and the ghosts finished their moves, so we should increase the depth
            if (successorAgent == numOfAgents):
                value = min(value, self.value(successor, 0, depth+1, alpha, beta))
            else:
                value = min(value, self.value(successor, successorAgent, depth, alpha, beta))

            if (value < alpha):
                return value

            beta = min(beta, value)

        return value

    def value(self, gameState, agent, depth, alpha, beta):
        # If we reach the leaf node
        if (gameState.isWin() or gameState.isLose() or self.depth == depth):
            return self.evaluationFunction(gameState)

        # If the agent is Pacman, calculate MAX. Else, calculate MIN.
        if (agent == 0):
            return self.maxVal(gameState, agent, depth, alpha, beta)
        else:
            return self.minVal(gameState, agent, depth, alpha, beta)

    def getBestAction(self, gameState, agent, depth):
        value = float('-inf')

        alpha = -float('inf')
        beta = float('inf')

        legalActions = gameState.getLegalActions(0)

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)

            succValue = self.value(successor, agent+1, depth, alpha, beta)

            if (succValue > value):
                value = succValue
                alpha = value
                act = action

        return act
    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        PACMAN = 0
        DEPTH = 0
        return self.getBestAction(gameState, PACMAN, DEPTH)

    def maxVal(self, gameState, agent, depth):
        value = float('-inf')

        legalActions = gameState.getLegalActions(agent)

        for action in legalActions:
            successor = gameState.generateSuccessor(agent,action)

            # Since only PACMAN calls this function, successorAgent will always be 1
            # But in the case of someone calls this function with the wrong agent, I added this part
            successorAgent = agent + 1

            if (successorAgent == 1):
                value = max(value, self.value(successor, successorAgent, depth))
            else:
                value = float('-inf')
        
        return value

    def value(self, gameState, agent, depth):
        # If we reach the leaf node
        if (gameState.isWin() or gameState.isLose() or self.depth == depth):
            return self.evaluationFunction(gameState)

        # If the agent is Pacman, calculate MAX. Else, calculate MIN.
        if (agent == 0):
            return self.maxVal(gameState, agent, depth)
        else:
            return self.ghostExpected(gameState, agent, depth)

    
    def ghostExpected(self, gameState, agent, depth):
        value = 0
        numOfAgents = gameState.getNumAgents()

        legalActions = gameState.getLegalActions(agent)

        succProbability = 1/len(legalActions)

        for action in legalActions:
            successor = gameState.generateSuccessor(agent,action)

            successorAgent = agent + 1

            # If the successor agent is PACMAN, then the agent index will 0 in the function call
            # Also, this means that PACMAN and the ghosts finished their moves, so we should increase the depth
            if (successorAgent == numOfAgents):
                value += succProbability * self.value(successor, 0, depth+1)
            else:
                value += succProbability * self.value(successor, successorAgent, depth)

        return value

    def getBestAction(self, gameState, agent, depth):
        value = float('-inf')
        legalActions = gameState.getLegalActions(0)

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)

            succValue = self.value(successor, agent+1, depth)

            if (succValue > value):
                value = succValue
                act = action

        return act

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information gathering 
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    foodList = food.asList()

    # Give reward and penalty to specific attributes
    score = 0
    minFoodDist = 999999
    minGhostDist = 999999
    numFood = len(foodList)

    ghostPenalty = -10
    losePenalty = -5000
    foodReward = 20
    eatGhostReward = 140
    ghostEffect = 0
    foodEffect = 0

    # Base value for the score like in the reflex agent
    score += currentGameState.getScore()

    if (len(foodList) == 0):
        foodEffect = 1000
    else:
        for f in foodList:
            minFoodDist = min(minFoodDist, manhattanDistance(position, f))

        foodEffect = (foodReward/(minFoodDist))


    i = 0
    for ghostState in ghostStates:
        currGhostDist = manhattanDistance(position, ghostState.getPosition())
        minGhostDist = min(minGhostDist, currGhostDist)

        if (currGhostDist < scaredTimes[i]):
            ghostEffect = eatGhostReward/(currGhostDist+0.1)
        else:
            if (minGhostDist == 0):
                return losePenalty

            ghostEffect = ghostPenalty/minGhostDist
        
        i += 1

    score = foodEffect + ghostEffect - 10*numFood

    return score


# Abbreviation
better = betterEvaluationFunction