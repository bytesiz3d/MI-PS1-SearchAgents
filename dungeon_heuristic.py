import enum
from dungeon import DungeonProblem, DungeonState
from mathutils import Direction, Point, euclidean_distance, manhattan_distance
from helpers import utils
from typing import Any, Dict

# This heuristic returns the distance between the player and the exit as an estimate for the path cost
# While it is consistent, it does a bad job at estimating the actual cost thus the search will explore a lot of nodes before finding a goal


def weak_heuristic(problem: DungeonProblem, state: DungeonState):
    return euclidean_distance(state.player, problem.layout.exit)


def strong_heuristic(problem: DungeonProblem, state: DungeonState) -> float:
    '''
    - `cache[points]` contains all remaining coins and the goal state
    '''
    cache = problem.cache()
    if "points" not in cache:
        cache_init_points(problem, cache)

        # print(f'$$ {len(problem.initial_state.remaining_coins)} LEN {len(cache["points"])}')
        # for pt, d in zip(cache["points"], cache["distances"]):
        #     print(pt, d)

    points = cache["points"]
    total_points = len(cache["points"])
    idx = (total_points - 1) - len(state.remaining_coins)

    return manhattan_distance(state.player, points[idx]) + cache["distances"][idx]


def cache_init_points(problem: DungeonProblem, cache: Dict[Any, Any]):
    '''
    - Sort points by closeness to each other
    - Form a path ending with the goal
    '''
    # Form a path of coins towards the exit
    cache["points"] = [problem.layout.exit]
    cache["distances"] = [0]
    initial_state = problem.get_initial_state()

    # Empty coins
    coins = list(initial_state.remaining_coins)
    if not len(coins): return

    while coins:
        closest_coin = 0
        closest_coin_dist = problem.layout.width * problem.layout.height
        for i, coin in enumerate(coins):
            dist = manhattan_distance(cache["points"][0], coin)
            if dist < closest_coin_dist:
                closest_coin = i
                closest_coin_dist = dist

        cache["points"].insert(0, coins[closest_coin])
        cache["distances"].insert(0, closest_coin_dist + cache["distances"][0])
        del coins[closest_coin]
