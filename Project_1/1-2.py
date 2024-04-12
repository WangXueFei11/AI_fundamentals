import sys

def dijkstra(graph, start):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    visited = [False] * n

    for _ in range(n):
        min_dist = float('inf')
        min_index = -1
        for i in range(n):
            if not visited[i] and dist[i] < min_dist:
                min_dist = dist[i]
                min_index = i

        if min_index == -1:
            return [-1] * n

        visited[min_index] = True

        for neighbor, weight in graph[min_index]:
            if not visited[neighbor] and dist[min_index] + weight < dist[neighbor]:
                dist[neighbor] = dist[min_index] + weight

    return dist

n, m = map(int, input().split())
graph = [[] for _ in range(n)]

for _ in range(m):
    x, y, z = map(int, input().split())
    graph[x - 1].append((y - 1, z))

dist = dijkstra(graph, 0)
if dist[n - 1] != float('inf'):
    print(dist[n - 1])
else:
    print(-1)
