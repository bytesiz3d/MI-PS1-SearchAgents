import enum
from dungeon import DungeonProblem, DungeonState
from mathutils import Direction, Point, euclidean_distance, manhattan_distance
from helpers import utils
from typing import Any, Dict, FrozenSet, List, Tuple, Callable

# This heuristic returns the distance between the player and the exit as an estimate for the path cost
# While it is consistent, it does a bad job at estimating the actual cost thus the search will explore a lot of nodes before finding a goal

DistanceFunction = Callable[[Point, Point], float]
distance_fn: DistanceFunction = manhattan_distance

def weak_heuristic(problem: DungeonProblem, state: DungeonState):
    return euclidean_distance(state.player, problem.layout.exit)


def strong_heuristic(problem: DungeonProblem, state: DungeonState) -> float:
    init_path(problem)
    goal, distance_to_exit = get_current_goal(problem, state.remaining_coins, state.player)
    return distance_fn(state.player, goal) + distance_to_exit


def get_current_goal(problem: DungeonProblem, remaining_coins: FrozenSet[Point], player: Point) -> Tuple[Point, int]:
    goals = problem.cache()["goals"]
    dists = problem.cache()["distances"]

    for i, goal in enumerate(goals):
        if goal in remaining_coins:
            return goal, dists[-1] - dists[i]

    return problem.layout.exit, 0


def init_path(problem: DungeonProblem):
    '''
    - Sort points by closeness to each other
    - Form a path ending with the goal
    '''
    # Form a path of coins towards the exit
    cache = problem.cache()
    goals = cache["goals"] = []
    dists = cache["distances"] = []

    last_point = problem.initial_state.player
    last_dist = 0

    # Empty coins
    initial_state = problem.get_initial_state()
    coins = list(initial_state.remaining_coins)
    if not len(coins): return

    coins = sorted(coins, key=lambda x: distance_fn(x, problem.layout.exit), reverse=True)
    while coins:
        closest_coin = -1
        closest_coin_dist = problem.layout.width * problem.layout.height

        for i, coin in enumerate(coins):
            dist = distance_fn(last_point, coin)
            if dist < closest_coin_dist:
                closest_coin = i
                closest_coin_dist = dist

        last_dist += closest_coin_dist
        last_point = coins[closest_coin]

        goals.append(last_point)
        dists.append(last_dist)
        del coins[closest_coin]
    
    last_dist += distance_fn(last_point, problem.layout.exit)
    last_point = problem.layout.exit
    goals.append(last_point)
    dists.append(last_dist)
