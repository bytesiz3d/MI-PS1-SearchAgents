import enum
from dungeon import DungeonProblem, DungeonState
from mathutils import Direction, Point, euclidean_distance, manhattan_distance
from helpers import utils
from typing import Any, Dict, FrozenSet, List, Tuple, Callable

# This heuristic returns the distance between the player and the exit as an estimate for the path cost
# While it is consistent, it does a bad job at estimating the actual cost thus the search will explore a lot of nodes before finding a goal

DistanceFunction = Callable[[Point, Point], float]
distance_fn: DistanceFunction = manhattan_distance

PHI: FrozenSet = frozenset()

def weak_heuristic(problem: DungeonProblem, state: DungeonState):
    return euclidean_distance(state.player, problem.layout.exit)


def strong_heuristic(problem: DungeonProblem, state: DungeonState) -> float:
    if "goals" not in problem.cache():
        init_path(problem)

    goal, distance_to_exit = get_current_goal(problem, state.remaining_coins, state.player)
    return distance_fn(state.player, goal) + distance_to_exit


def get_current_goal(problem: DungeonProblem, remaining_coins: FrozenSet[Point], player: Point) -> Tuple[Point, int]:
    goals = problem.cache()["goals"]
    dists = problem.cache()["distances"]
    min_superset = remaining_coins

    if min_superset not in goals:
        min_superset = list(goals)[0]
        for superset in goals:
            if superset > remaining_coins and superset < min_superset:
                min_superset = superset


    return goals[min_superset], dists[PHI] - dists[min_superset]


def init_path(problem: DungeonProblem):
    '''
    - Sort points by closeness to each other
    - Form a path ending with the goal
    '''
    # Form a path of coins towards the exit
    goals = problem.cache()["goals"] = {}
    dists = problem.cache()["distances"] = {}

    last_point = problem.initial_state.player
    last_dist = 0

    # Empty coins
    initial_state = problem.get_initial_state()
    coins = sorted(
        initial_state.remaining_coins, 
        key = lambda x: distance_fn(x, problem.layout.exit),
        reverse = True
    )
    while coins:
        closest_coin = min(coins, key=lambda pt: distance_fn(last_point, pt))
        closest_coin_dist = distance_fn(last_point, closest_coin)

        last_dist += closest_coin_dist
        last_point = closest_coin

        fs = frozenset(coins)
        goals[fs] = last_point
        dists[fs] = last_dist
        coins.remove(closest_coin)
    
    last_dist += distance_fn(last_point, problem.layout.exit)
    last_point = problem.layout.exit
    goals[PHI] = last_point
    dists[PHI] = last_dist
