"""Simple grid-based Dynamic Window Approach (DWA) helper.

This is a lightweight grid-compatible approximation of DWA used to pick an
alternative next cell when the planned next waypoint is occupied by a dynamic
obstacle. The function `choose_alternative` scores neighbor candidate cells by
progress towards the goal and clearance from predicted dynamic obstacle cells.

API:
- choose_alternative(robot_rc, goal_rc, grid, dynamic_obstacles) -> (r,c) or None
"""
from typing import Tuple, List, Optional
import math


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def predicted_unsafe_cells(dynamic_obstacles, rows, cols):
    """Return a set of cells (r,c) that are likely unsafe in the near future.
    We conservatively include each dynamic obstacle's cell and its 4-neighbors.
    """
    unsafe = set()
    for d in dynamic_obstacles:
        rr = d.get('r')
        rc = d.get('c')
        if rr is None or rc is None:
            continue
        for dr, dc in ((0,0),(-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = rr + dr, rc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                unsafe.add((nr, nc))
    return unsafe


def choose_alternative(robot_rc: Tuple[int,int], goal_rc: Tuple[int,int], grid: List[List[object]], dynamic_obstacles: List[dict]) -> Optional[Tuple[int,int]]:
    """Choose a neighbor cell to avoid dynamic obstacles.

    robot_rc: (r,c) current robot cell
    goal_rc: (r,c) the current high-level goal (may be nearest dirt)
    grid: 2D list of Cell objects
    dynamic_obstacles: list of dicts with 'r','c' fields

    Returns a neighbor (r,c) to move to, or None if no safe alternative found.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    rr, rc = robot_rc
    # candidate moves: stay or 4-neighbors
    candidates = [(rr, rc), (rr-1, rc), (rr+1, rc), (rr, rc-1), (rr, rc+1)]

    unsafe = predicted_unsafe_cells(dynamic_obstacles, rows, cols)

    best = None
    best_score = -math.inf
    for (r,c) in candidates:
        if not (0 <= r < rows and 0 <= c < cols):
            continue
        cell = grid[r][c]
        # cannot move into static barrier or dynamic cell right now
        if cell.is_barrier or getattr(cell, 'has_dynamic', False):
            continue
        # compute clearance to predicted dynamic cells (manhattan)
        min_dist = min(manhattan((r,c), u) for u in unsafe) if unsafe else 999
        # progress to goal (we want smaller distance)
        dist_goal = manhattan((r,c), goal_rc) if goal_rc is not None else 0
        # scoring: prefer larger clearance and closer to goal
        score = (-dist_goal) + 0.7 * min_dist
        # slight tie-breaker: prefer not staying in place
        if (r,c) == (rr,rc):
            score -= 0.1
        if score > best_score:
            best_score = score
            best = (r,c)

    # if best is just staying in place and it's unsafe (in unsafe set), return None
    return None if best == (rr,rc) and (rr,rc) in unsafe else best
