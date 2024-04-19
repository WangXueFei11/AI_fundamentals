import matplotlib.pyplot as plt

def dfs(n, m, maze):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    stack = [(0, 0)]
    parent = {(0, 0): None}
    search_path = []
    while stack:
        row, col = stack.pop()
        search_path.append((row, col))

        if row == n - 1 and col == m - 1:
            path = []
            while (row, col) != (0, 0):
                path.append((row, col))
                row, col = parent[(row, col)]
            path.append((0, 0))
            path.reverse()
            return len(path) - 1, path, search_path

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if 0 <= new_row < n and 0 <= new_col < m and maze[new_row][new_col] == 0 and (new_row, new_col) not in parent:
                parent[(new_row, new_col)] = (row, col)
                stack.append((new_row, new_col))

    return -1, [], search_path

def visualize_maze_with_path(maze, path, search_paths):
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

(result, path, search_path) = dfs(n, m, maze)
print(result)
print("Path:", path)
print("search_path:", search_path)
visualize_maze_with_path(maze, path, search_path)
