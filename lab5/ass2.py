from typing import List, Tuple
import heapq

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def neighbors(grid: List[List[str]], r: int, c: int) -> List[Tuple[int, int]]:
    """Return available neighbors of cell (r, c)"""
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    result = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] not in ['@', 'T']:
            result.append((nr, nc))
    return result

def a_star_search(grid: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int], pushing: bool) -> int:
    """Return shortest path from start to goal. If pushing=True, consider pushing $"""
    open_list = [(manhattan_distance(start, goal), 0, start)]
    visited = set()

    while open_list:
        _, cost, current = heapq.heappop(open_list)
        if current == goal:
            return cost
        if current in visited:
            continue
        visited.add(current)

        for neighbor in neighbors(grid, current[0], current[1]):
            if pushing and grid[neighbor[0]][neighbor[1]] == '$':
                # Determine the direction of push
                dr, dc = neighbor[0] - current[0], neighbor[1] - current[1]
                new_r, new_c = neighbor[0] + dr, neighbor[1] + dc
                if 0 <= new_r < len(grid) and 0 <= new_c < len(grid[0]) and grid[new_r][new_c] == '#':
                    grid[neighbor[0]][neighbor[1]] = '#'  # Empty the spot where the item was
                    grid[new_r][new_c] = '$'  # Place the item in the new position
                    heapq.heappush(open_list, (manhattan_distance((new_r, new_c), goal) + cost + 1, cost + 1, (new_r, new_c)))
            else:
                heapq.heappush(open_list, (manhattan_distance(neighbor, goal) + cost + 1, cost + 1, neighbor))
    return float('inf')



def solve(grid: List[List[str]]) -> int:
    robot_position = None
    item_position = None
    shelf_positions = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 'R':
                robot_position = (r, c)
            elif grid[r][c] == '$':
                item_position = (r, c)
            elif grid[r][c] == 'T':
                shelf_positions.append((r, c))

    min_pushes = float('inf')
    for shelf_position in shelf_positions:
        # Check robot reaching the item
        to_item = a_star_search(grid, robot_position, item_position, False)
        # Check robot pushing the item to the shelf
        to_shelf = a_star_search(grid, item_position, shelf_position, True)
        min_pushes = min(min_pushes, to_item + to_shelf)

    if min_pushes == float('inf'):
        return -1
    return min_pushes

# Testing
grid1 = [["@", "@", "@", "@", "@", "@"],
         ["@", "@", "@", "@", "T", "@"],
         ["@", "#", "$", "#", "#", "@"],
         ["@", "#", "@", "@", "#", "@"],
         ["@", "R", "#", "#", "#", "@"],
         ["@", "T", "@", "@", "@", "@"]]
print(solve(grid1))  # Expected: 3

grid2 = [["@", "T", "@", "@", "@", "@"],
         ["@", "#", "@", "@", "@", "@"],
         ["@", "#", "#", "#", "$", "@"],
         ["@", "#", "@", "@", "@", "@"],
         ["@", "R", "#", "#", "T", "@"],
         ["@", "@", "@", "@", "@", "@"]]
print(solve(grid2))  # Expected: -1

# test case 3
grid3 = [["@", "T", "@", "@", "@", "@"],
         ["@", "#", "@", "@", "@", "#"],
         ["@", "#", "#", "#", "#", "$"],
         ["@", "#", "@", "@", "@", "#"],
         ["@", "R", "#", "#", "T", "@"],
         ["@", "@", "@", "@", "@", "@"]]
print(solve(grid3))

