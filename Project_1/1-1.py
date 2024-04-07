from collections import deque

def bfs(graph, start, end):
    queue = deque([start])
    visited = [False] * (len(graph) + 1)
    visited[start] = True
    distance = [float('inf')] * (len(graph) + 1)
    distance[start] = 0
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if not visited[neighbor]:
                queue.append(neighbor)
                visited[neighbor] = True
                distance[neighbor] = distance[current] + 1

    return distance[end] if distance[end] != float('inf') else -1

n, m = map(int, input().split())
graph = [[] for _ in range(n + 1)]
for _ in range(m):
    a, b = map(int, input().split())
    graph[a].append(b)

print(bfs(graph, 1, n))
