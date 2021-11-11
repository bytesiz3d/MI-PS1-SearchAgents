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
    explored: bool = False

    def __lt__(self, other: Meta) -> bool:
        return self.total_cost < other.total_cost


class ISearch:
    '''### ISearch
    General search agent interface. Implements the standard search procedure.

    Each algorithm is represented as a class which overrides abstract elements of the search procedure.
    '''
    # Frontier data structure
    frontier: Any

    # Adds a state to the frontier
    add: Callable[[S], None]

    # Retrieves a state from the frontier
    retrieve: Callable[[], Tuple[Meta, S]]

    # Given a state and its total cost, returns if it should be added to the frontier
    should_add: Callable[[S, float], bool]

    # Returns Tuple(total cost, backward cost) given a state, an action, and its successor
    get_cost: Callable[[Problem, HeuristicFunction, float, S, A, S], Tuple[float, float]] = lambda *_: (0, 0)

    # Cache of state to metadata
    cache: Dict[S, Meta] = {}

    def trace_path(self, state: S) -> Solution:
        '''
        Traces the path from the goal state to the start state, returns a list of actions.
        '''
        path = []
        meta = self.cache[state]

        while meta.parent:
            path.append(meta.action)
            meta = self.cache[meta.parent]

        return path[::-1]

    def search(self, problem: Problem[S, A], initial_state: S, heuristic: HeuristicFunction = None) -> Solution:
        '''
        Implements the standard search algorithm, returns a list of actions if a path is found,
        and `None` otherwise.
        '''
        self.cache[initial_state] = Meta()
        self.add(initial_state)

        while self.frontier:
            metadata, state = self.retrieve()
            backward_cost = metadata.backward_cost

            if self.cache[state].explored: continue
            self.cache[state].explored = True

            if problem.is_goal(state): return self.trace_path(state)

            actions = problem.get_actions(state)
            for s_act in actions:
                successor = problem.get_successor(state, s_act)
                total, back = self.get_cost(problem, heuristic, backward_cost, state, s_act, successor)

                if self.should_add(successor, total):
                    self.cache[successor] = Meta(state, s_act, total, back)
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
        self.cache = {}

    def add(self, state: S):
        self.frontier.append((self.cache[state], state))

    def should_add(self, state: S, cost: float) -> bool:
        return state not in self.cache


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
        self.cache = {}

    def add(self, state: S):
        self.frontier.appendleft((self.cache[state], state))

    def should_add(self, state: S, cost: float) -> bool:
        first = state not in self.cache
        return first or not self.cache[state].explored


class IPrioritySearch(ISearch):
    '''### IPrioritySearch
    Cost-based Search
        - Frontier is a priority queue
        - States are added to the frontier whenever they're found with a lower cost
    '''

    def __init__(self):
        self.frontier = list()
        self.cache = {}

    def retrieve(self) -> Tuple[Meta, S]:
        return heappop(self.frontier)

    def add(self, state: S):
        heappush(self.frontier, (self.cache[state], state))

    def should_add(self, state: S, cost: float) -> bool:
        first = state not in self.cache
        return first or not self.cache[state].explored and cost < self.cache[state].total_cost


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