import heapq

def dijkstra(graph, start, end):
    n = len(graph)
    distance = [float('inf')] * n
    distance[start] = 0
    pq = [(0, start)]
    visited = set()

    while pq:
        curr_dist, curr_node = heapq.heappop(pq)
        if curr_node in visited:
            continue
        visited.add(curr_node)
        if curr_node == end:
            return curr_dist
        for neighbor, weight in graph[curr_node]:
            new_dist = curr_dist + weight
            if new_dist < distance[neighbor]:
                distance[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    return -1

n, m = map(int, input().split())
graph = [[] for _ in range(n)]
for _ in range(m):
    x, y, z = map(int, input().split())
    graph[x - 1].append((y - 1, z))
start_node = 0
end_node = n - 1
print(dijkstra(graph, start_node, end_node))
