from typing import List, Tuple
import heapq


def solve(grid: List[List[str]]) -> int:
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up

    robot = item = None
    shelves = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 'R':
                robot = (i, j)
            elif grid[i][j] == '$':
                item = (i, j)
            elif grid[i][j] == 'T':
                shelves.append((i, j))

    def is_adjacent(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1

    def can_push(robot, item, dx, dy):
        new_item = (item[0] + dx, item[1] + dy)
        return (is_adjacent(robot, item) and
                0 <= new_item[0] < rows and
                0 <= new_item[1] < cols and
                grid[new_item[0]][new_item[1]] == '#')

    def manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def a_star_search(start_robot, start_item, target):
        visited = set()
        pq = [(manhattan(start_item, target), 0, start_robot, start_item)]  # heuristic, pushes, robot_pos, item_pos

        while pq:
            _, pushes, robot, item = heapq.heappop(pq)

            if item == target:
                return pushes

            if (robot, item) in visited:
                continue
            visited.add((robot, item))

            for dx, dy in directions:
                new_r = (robot[0] + dx, robot[1] + dy)
                if 0 <= new_r[0] < rows and 0 <= new_r[1] < cols and grid[new_r[0]][new_r[1]] not in ('@', '$'):
                    if can_push(robot, item, dx, dy):
                        new_item = (item[0] + dx, item[1] + dy)
                        heapq.heappush(pq, (pushes + 1 + manhattan(new_item, target), pushes + 1, new_r, new_item))
                    else:
                        heapq.heappush(pq, (pushes + manhattan(item, target), pushes, new_r, item))

        return float('inf')  # If no path exists

    pushes_to_shelves = [a_star_search(robot, item, shelf) for shelf in shelves]
    min_pushes = min(pushes_to_shelves, default=float('inf'))

    return -1 if min_pushes == float('inf') else min_pushes


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

