"""Simple grid-based search algorithms: A* and Dijkstra (uniform-cost).
API:
- astar(start, goal, grid) -> list of (r,c) path from start to goal (inclusive) or [] if no path
- dijkstra(start, goal, grid) -> same

Grid is a 2D list of cells where cell.is_barrier indicates impassable.
Start/goal are (r, c) tuples.
"""
from heapq import heappush, heappop


def neighbors(pos, rows, cols):
    r, c = pos
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def heuristic(a, b):
    # Manhattan
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start, goal, grid):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}

    while open_set:
        _, current = heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)
        for n in neighbors(current, rows, cols):
            nr, nc = n
            # treat dynamic obstacles as barriers for planning too
            if grid[nr][nc].is_barrier or getattr(grid[nr][nc], 'has_dynamic', False):
                continue
            tentative_g = gscore[current] + 1
            if tentative_g < gscore.get(n, float('inf')):
                came_from[n] = current
                gscore[n] = tentative_g
                fscore[n] = tentative_g + heuristic(n, goal)
                heappush(open_set, (fscore[n], n))
    return []


def dijkstra(start, goal, grid):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    pq = []
    heappush(pq, (0, start))
    dist = {start: 0}
    came_from = {}

    while pq:
        d, current = heappop(pq)
        if current == goal:
            return reconstruct_path(came_from, current)
        if d > dist.get(current, float('inf')):
            continue
        for n in neighbors(current, rows, cols):
            nr, nc = n
            # treat dynamic obstacles as barriers for planning too
            if grid[nr][nc].is_barrier or getattr(grid[nr][nc], 'has_dynamic', False):
                continue
            nd = d + 1
            if nd < dist.get(n, float('inf')):
                dist[n] = nd
                came_from[n] = current
                heappush(pq, (nd, n))
    return []
