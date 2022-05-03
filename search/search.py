# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # Idea: Expand the deepest node first
    # Frontier is a LIFO Stack

    startState = problem.getStartState()
    path = []

    frontier = util.Stack()
    frontier.push((startState, path))

    visitedStates = set()

    while not frontier.isEmpty():
        newState, path = frontier.pop()

        if newState in visitedStates:
            continue

        visitedStates.add(newState)
        if problem.isGoalState(newState):
            return path
        
        successors = problem.getSuccessors(newState)

        for successorState,action,stepCost in successors:
            if successorState not in visitedStates:
                newPath = path + [action]
                frontier.push((successorState, newPath))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Idea: Expand the shallowest node first
    # Frontier is a FIFO queue

    startState = problem.getStartState()
    path = []

    frontier = util.Queue()
    frontier.push((startState, path))

    visitedStates = set()

    while not frontier.isEmpty():
        newState, path = frontier.pop()
        #print(newState)

        if newState in visitedStates:
            continue

        visitedStates.add(newState)
        if problem.isGoalState(newState):
            return path
        
        successors = problem.getSuccessors(newState)

        for successorState,action,stepCost in successors:
            if successorState not in visitedStates:
                newPath = path + [action]
                frontier.push((successorState, newPath))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Idea: Expand the lowest cost node first
    # Frontier is a priority queue based on cumulative cost

    startState = problem.getStartState()
    path = []

    frontier = util.PriorityQueue()
    frontier.push((startState, path), 0)

    visitedStates = set()

    while not frontier.isEmpty():
        newState, path = frontier.pop()

        if newState in visitedStates:
            continue

        visitedStates.add(newState)
        if problem.isGoalState(newState):
            return path
        
        successors = problem.getSuccessors(newState)
        previousCost = problem.getCostOfActions(path)

        for successorState,action,stepCost in successors:
            if successorState not in visitedStates:
                newPath = path + [action]
                cumulativeCost = previousCost + stepCost
                frontier.push((successorState, newPath), cumulativeCost)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Idea: Expand the nodes that is both low cost and promising
    # Frontier is a priority queue based on cumulative cost + heuristic (f(s) = g(s)+h(s))

    startState = problem.getStartState()
    path = []

    frontier = util.PriorityQueue()
    frontier.push((startState, path), 0)

    visitedStates = set()

    while not frontier.isEmpty():
        newState, path = frontier.pop()

        if newState in visitedStates:
            continue

        visitedStates.add(newState)
        if problem.isGoalState(newState):
            return path
        
        successors = problem.getSuccessors(newState)
        previousCost = problem.getCostOfActions(path)

        for successorState,action,stepCost in successors:
            if successorState not in visitedStates:
                newPath = path + [action]
                # f(n) = g(n) + h(n) and g(n) = g(n-1) + cost(from n-1 to n)
                cumulativeCost = previousCost + stepCost + heuristic(successorState, problem)
                frontier.push((successorState, newPath), cumulativeCost)
    


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
