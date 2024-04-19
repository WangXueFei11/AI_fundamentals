import heapq
import matplotlib.pyplot as plt

def heuristic(start, goal):
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])   # 曼哈顿距离为启发函数
def Astar(n, m, maze):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dist = {(i, j): float('inf') for i in range(n) for j in range(m)}
    dist[(0, 0)] = 0
    pq = [(0 + heuristic((0, 0), (m - 1, n - 1)), (0, 0))]
    search_path = [(0, 0)]
    parent = {}
    while pq:
        f, (row, col) = heapq.heappop(pq)
        g = f - heuristic((row, col), (m - 1, n - 1))
        if row == n - 1 and col == m - 1:
            path = []
            while (row, col) != (0, 0):
                path.append((row, col))
                row, col = parent[(row, col)]
            path.append((0, 0))
            path.reverse()
            return g, path, search_path
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < n and 0 <= new_col < m:
                new_dist = g + 1
                if new_dist < dist[(new_row, new_col)] and maze[new_row][new_col] == 0:
                    dist[(new_row, new_col)] = new_dist
                    heapq.heappush(pq, (new_dist + heuristic((new_row, new_col), (m - 1, n - 1)), (new_row, new_col)))
                    search_path.append((new_row, new_col))
                    parent[(new_row, new_col)] = (row, col)
    return -1, [], set()

def visualize_maze_with_path(maze, path, search_path):
    plt.figure(figsize=(len(maze[0]), len(maze)))
    plt.imshow(maze, cmap='Greys', interpolation='nearest')
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, marker='o', markersize=8, color='red', linewidth=3)
    for tuple_s in search_path:
        plt.fill([tuple_s[1]-0.5, tuple_s[1] + 0.5, tuple_s[1] + 0.5, tuple_s[1]-0.5],
                 [tuple_s[0]-0.5, tuple_s[0]-0.5, tuple_s[0] + 0.5, tuple_s[0] + 0.5],
                 color='grey')

    plt.xticks(range(len(maze[0])))
    plt.yticks(range(len(maze)))
    plt.gca().set_xticks([x - 0.5 for x in range(1, len(maze[0]))], minor=True)
    plt.gca().set_yticks([y - 0.5 for y in range(1, len(maze))], minor=True)
    plt.grid(which="minor", color="black", linestyle='-', linewidth=2)
    plt.axis('on')
    plt.show()

n, m = map(int, input().split())
maze = [list(map(int, input().split())) for _ in range(n)]

result, path, search_path = Astar(n, m, maze)
print(result)
print('path:',path)
print('search_path:',search_path)
visualize_maze_with_path(maze, path, search_path)
