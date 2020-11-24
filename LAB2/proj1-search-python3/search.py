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
    #création de la pile
    fringe = util.Stack()  #last-in-first-out (LIFO)
    fringe.push( (problem.getStartState(), [], []) )
    explored = set()
    #parcours la pile tant qu'elle est pas vide 
    while fringe is not fringe.isEmpty():
        #tant que la boucle "tourne" on enleve les éléments inutiles de la pile
        current_node, actions, visited = fringe.pop()
        #on verifie qu'on est pas passé par le point d'arrivé (si c'est le cas on renvoit le chemin)
        if problem.isGoalState(current_node):
            return path
        #sinon on rejoute dans la liste les éléments nécessaires pour continuer (localisation, action..)
        if current_node not in explored:
            for neighbor_location, direction, steps in problem.getSuccessors(current_node):        
                fringe.push((neighbor_location, actions + [direction] , visited + [current_node] ))
                path = actions + [direction]
                explored.add(current_node)                       
    return []
    util.raiseNotDefined()
     
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #création de la pile
    fringe = util.Queue()  #first-in-first-out (FIFO)
    fringe.push((problem.getStartState(),[],[]))
    #créatin d'une liste vide 
    explored = []
    #parcours la pile tant qu'elle est pas vide 
    while fringe is not fringe.isEmpty():
        #tant que la boucle "tourne" on enleve les éléments inutiles de la pile
        current_node, actions, Cost = fringe.pop()     
        if current_node not in explored:
            explored.append(current_node)
            #on verifie qu'on est pas passé par le point d'arrivé 
            if problem.isGoalState(current_node):
                return actions
            #sinon on rejoute dans la liste les éléments nécessaires pour continuer (localisation, action..)
            for neighbor_location, direction, cost in problem.getSuccessors(current_node):
                fringe.push((neighbor_location, actions+[direction], Cost + [cost]))

    return []
    util.raiseNotDefined()
    
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #création de la pile
    fringe = util.PriorityQueue()
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    fringe.push( (problem.getStartState(), [], 0), 0 )
    #créatin d'une liste vide 
    explored = []
    #parcours la pile tant qu'elle est pas vide 
    while not fringe.isEmpty():
        #tant que la boucle "tourne" on enleve les éléments inutiles de la pile
         current_node, actions, Cost = fringe.pop()
         if current_node not in explored:
            explored.append(current_node)
            #on verifie qu'on est pas passé par le point d'arrivé
            if problem.isGoalState(current_node):
                return actions
            #sinon on rejoute dans la liste les éléments nécessaires pour continuer (localisation, action..)
            for neighbor_location, direction, cost in problem.getSuccessors(current_node):
                fringe.push((neighbor_location, actions+[direction], Cost + cost), Cost + cost)
    return []
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #création de la pile
    fringe = util.PriorityQueue()
    #créatin d'une liste vide 
    explored = {} 
    #verif. si on est a l'état final (si c'est le cas on renvoie le resultat)
    if problem.isGoalState(problem.getStartState()):
        return []
    fringe.push((problem.getStartState(),[]),0)
    #parcours la pile tant qu'elle est pas vide 
    while fringe is not fringe.isEmpty():
        #tant que la boucle "tourne" on enleve les éléments inutiles de la pile
        Current_State, path = fringe.pop()
        cCost = problem.getCostOfActions(path)
        #verif. si on est a l'état final (si c'est le cas on renvoie le resultat)
        if problem.isGoalState(Current_State):
            return path
        #sinon on explore pour trouver l'état final
        if Current_State not in explored or cCost<explored[Current_State]:
            explored[Current_State]=cCost
            for successor,action,stepCost in problem.getSuccessors(Current_State):
                currentTotalCost = cCost + stepCost + heuristic(successor,problem)
                fringe.push((successor, path+[action]),currentTotalCost)
    return []
            


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
