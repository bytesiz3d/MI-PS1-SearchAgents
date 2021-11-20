from __future__ import annotations
from argparse import Action
from typing import Any, Dict, Callable, Tuple
from agents import InformedSearchAgent
from problem import HeuristicFunction, Problem, S, A, Solution
from collections import deque
from helpers import utils
from dataclasses import dataclass, field
from heapq import heappush, heappop

# TODO: Import any modules or write any helper functions you want to use

# All search functions take a problem and a state
# If it is an informed search function, it will also receive a heuristic function
# S and A are used for generic typing where S represents the state type and A represents the action type

# All the search functions should return one of two possible type:
# 1. A list of actions which represent the path from the initial state to the final state
# 2. None if there is no solution


@dataclass
class Meta:
    '''### Meta
    Represents metadata for states. This is used to trace back the paths each node took,
    as well as their backward costs.
    '''
    parent: S = None
    action: A = None
    total_cost: float = 0
    backward_cost: float = 0
    order: int = 0
    explored: bool = False

    def __lt__(self, other: Meta) -> bool:
        '''
        Returns true if this state should precede another state.
        Maintains a stable order for the heapq by keeping insertion order.
        '''
        if self.total_cost == other.total_cost:
            return self.order < other.order

        return self.total_cost < other.total_cost


class ISearch:
    '''### ISearch
    General search agent interface. Implements the standard search procedure.

    Each algorithm is represented as a class which overrides abstract elements of the search procedure.
    '''
    # Frontier data structure
    frontier: Any

    # Adds a state to the frontier
    add: Callable[[S]]

    # Retrieves a state from the frontier
    retrieve: Callable[[], Tuple[Meta, S]]

    # Given a state and its total cost, returns if it should be added to the frontier
    should_add: Callable[[S, float], bool]

    # Returns Tuple(total cost, backward cost) given a state, an action, and its successor
    get_cost: Callable[[Problem, HeuristicFunction, float, S, A, S], Tuple[float, float]] = lambda *_: (0, 0)

    # Cache of state to metadata
    path: Dict[S, Meta]

    def trace_path(self, state: S) -> Solution:
        '''
        Traces the path from the goal state to the start state, returns a list of actions.
        '''
        action_path = []
        meta = self.path[state]

        while meta.parent:
            action_path.append(meta.action)
            meta = self.path[meta.parent]

        return action_path[::-1]

    def search(self, problem: Problem[S, A], initial_state: S, heuristic: HeuristicFunction = None) -> Solution:
        '''
        Implements the standard search algorithm, returns a list of actions if a path is found,
        and `None` otherwise.
        '''
        self.path = problem.cache()["path"] = {}

        # Increasing insertion order
        order = 0

        # Initial state has no cost or parent
        self.path[initial_state] = Meta()
        self.add(initial_state)

        # While the frontier contains nodes to expand
        while self.frontier:
            # Retrieve a node from the frontier
            metadata, state = self.retrieve()
            problem.cache()["current_parent"] = state
            backward_cost = metadata.backward_cost

            # Expand the node if it has not yet been expanded
            if self.path[state].explored: continue
            self.path[state].explored = True

            # Check if we have arrived at the goal
            if problem.is_goal(state): return self.trace_path(state)

            actions = problem.get_actions(state)
            for s_act in actions:
                successor = problem.get_successor(state, s_act)
                total, back = self.get_cost(problem, heuristic, backward_cost, state, s_act, successor)

                if self.should_add(successor, total):
                    order += 1
                    self.path[successor] = Meta(state, s_act, total, back, order)
                    self.add(successor)

        return None


class BFS(ISearch):
    '''### Breadth-first Search
        - Frontier is a queue
        - No cost is used in calculations
        - States are added to the frontier if they haven't been added already
    '''

    def __init__(self):
        self.frontier = deque()
        self.retrieve = self.frontier.popleft
        self.path = {}

    def add(self, state: S):
        self.frontier.append((self.path[state], state))

    def should_add(self, state: S, cost: float) -> bool:
        return state not in self.path


class DFS(ISearch):
    '''### Depth-first Search
        - Frontier is a stack
        - No cost is used in calculations
        - States are added to the frontier if they haven't been added already,
          or if they haven't been explored yet. This handles the case that a state is
          found multiple times further down the search tree.
    '''

    def __init__(self):
        self.frontier = deque()
        self.retrieve = self.frontier.popleft
        self.path = {}

    def add(self, state: S):
        self.frontier.appendleft((self.path[state], state))

    def should_add(self, state: S, cost: float) -> bool:
        first = state not in self.path
        return first or not self.path[state].explored


class IPrioritySearch(ISearch):
    '''### IPrioritySearch
    Cost-based Search
        - Frontier is a priority queue
        - States are added to the frontier whenever they're found with a lower cost
    '''

    def __init__(self):
        self.frontier = list()
        self.path = {}

    def retrieve(self) -> Tuple[Meta, S]:
        return heappop(self.frontier)

    def add(self, state: S):
        heappush(self.frontier, (self.path[state], state))

    def should_add(self, state: S, cost: float) -> bool:
        first = state not in self.path
        return first or not self.path[state].explored and cost < self.path[state].total_cost


class UCS(IPrioritySearch):
    '''### Uniform Cost Search
        - States with minimum backward cost are chosen
    '''

    def __init__(self):
        super().__init__()

    def get_cost(self, problem: Problem, heuristic: HeuristicFunction, backward_cost: float, state: S, action: A, successor: S):
        backward_cost += problem.get_cost(state, action)
        return backward_cost, backward_cost


class AStar(IPrioritySearch):
    '''### A* Search
        - States with minimum backward cost + heuristic are chosen
    '''

    def __init__(self):
        super().__init__()

    def get_cost(self, problem: Problem, heuristic: HeuristicFunction, backward_cost: float, state: S, action: A, successor: S):
        backward_cost += problem.get_cost(state, action)
        total_cost = backward_cost + heuristic(problem, successor)
        return total_cost, backward_cost


class GBFS(IPrioritySearch):
    '''### Greedy Best-First Search
        - States with minimum heuristic are chosen
    '''

    def __init__(self):
        super().__init__()

    def get_cost(self, problem: Problem, heuristic: HeuristicFunction, backward_cost: float, state: S, action: A, successor: S):
        total_cost = heuristic(problem, successor)
        return total_cost, 0


def BreadthFirstSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    agent = BFS()
    return agent.search(problem, initial_state)


def DepthFirstSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    agent = DFS()
    return agent.search(problem, initial_state)


def UniformCostSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    agent = UCS()
    return agent.search(problem, initial_state)


def AStarSearch(problem: Problem[S, A], initial_state: S, heuristic: HeuristicFunction) -> Solution:
    agent = AStar()
    return agent.search(problem, initial_state, heuristic)


def BestFirstSearch(problem: Problem[S, A], initial_state: S, heuristic: HeuristicFunction) -> Solution:
    agent = GBFS()
    return agent.search(problem, initial_state, heuristic)
