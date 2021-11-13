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
    goal, distance_to_exit = get_current_goal(problem, state.remaining_coins)

    # initial_coins = problem.initial_state.remaining_coins
    # eaten = state.player

    # # Verify if we're following the path
    # initial_coins = problem.initial_state.remaining_coins
    # eaten = state.player
    # if eaten in initial_coins:
    #     goal, distance_to_exit = get_current_goal(problem, state.remaining_coins | {eaten})

    #     # If path is being followed, revert check
    #     if goal == eaten:
    #         goal, distance_to_exit = get_current_goal(problem, state.remaining_coins)

    # print(state.player, state.remaining_coins, goal)
    # for rem, (pt, d) in problem.cache().items():
    #     print(rem, pt, d)
    # print("-" * 120)

    return manhattan_distance(state.player, goal) + distance_to_exit


def get_current_goal(problem: DungeonProblem, remaining_coins: FrozenSet[Point]) -> Tuple[Point, int]:
    cache = problem.cache()
    if remaining_coins in cache:
        return cache[remaining_coins]

    points = [problem.layout.exit]
    distances = [0]

    coins = list(remaining_coins)

    # Empty coins
    if not len(coins):
        cache[remaining_coins] = (points[0], distances[0])
        return cache[remaining_coins]

    while coins:
        # Find the closest coin to the last inserted coin
        closest_coin = 0
        closest_coin_dist = problem.layout.width * problem.layout.height
        for i, coin in enumerate(coins):
            dist = manhattan_distance(points[0], coin)
            if dist < closest_coin_dist:
                closest_coin = i
                closest_coin_dist = dist

        # Insert the coin and its distance
        points.insert(0, coins[closest_coin])
        distances.insert(0, closest_coin_dist + distances[0])

        # Add this state to cache
        remaining = frozenset(points[:-1])
        if remaining not in cache:
            cache[remaining] = (points[0], distances[0])

        del coins[closest_coin]

    return cache[remaining_coins]
