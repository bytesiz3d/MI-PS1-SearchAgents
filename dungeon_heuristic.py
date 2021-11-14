import enum
from dungeon import DungeonProblem, DungeonState
from mathutils import Direction, Point, euclidean_distance, manhattan_distance
from helpers import utils
from typing import Any, Dict, FrozenSet, List, Tuple

# This heuristic returns the distance between the player and the exit as an estimate for the path cost
# While it is consistent, it does a bad job at estimating the actual cost thus the search will explore a lot of nodes before finding a goal


def weak_heuristic(problem: DungeonProblem, state: DungeonState):
    return euclidean_distance(state.player, problem.layout.exit)


def strong_heuristic(problem: DungeonProblem, state: DungeonState) -> float:
    init_path(problem)
    goal, distance_to_exit = get_current_goal(problem, state.remaining_coins, state.player)
    return manhattan_distance(state.player, goal) + distance_to_exit


def get_current_goal(problem: DungeonProblem, remaining_coins: FrozenSet[Point], player: Point) -> Tuple[Point, int]:
    cache = problem.cache()
    points = cache["points"]

    for point in points:
        if point in remaining_coins:
            return point, manhattan_distance(problem.layout.exit, point)

    return problem.layout.exit, 0


def init_path(problem: DungeonProblem):
    '''
    - Sort points by closeness to each other
    - Form a path ending with the goal
    '''
    # Form a path of coins towards the exit
    cache = problem.cache()
    cache["points"] = []
    last_point = problem.initial_state.player

    # Empty coins
    initial_state = problem.get_initial_state()
    coins = list(initial_state.remaining_coins)
    if not len(coins): return

    while coins:
        closest_coin = 0
        closest_coin_dist = problem.layout.width * problem.layout.height

        for i, coin in enumerate(coins):
            dist = manhattan_distance(last_point, coin)
            if dist < closest_coin_dist:
                closest_coin = i
                closest_coin_dist = dist

        cache["points"].append(coins[closest_coin])
        last_point = coins[closest_coin]
        del coins[closest_coin]
