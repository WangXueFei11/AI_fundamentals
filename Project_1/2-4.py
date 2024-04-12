import heapq

def is_target(state):
    return state == [1, 2, 3, 4, 5, 6, 7, 8, 'x']

def heuristic(state):
    target = [1, 2, 3, 4, 5, 6, 7, 8, 'x']
    return sum(1 for a, b in zip(state, target) if a != b)

def get_neighbors(state, move):
    neighbors = []
    x_pos = state.index('x')
    x_row, x_col = divmod(x_pos, 3)

    directions = [(-1, 0, 'u'), (1, 0, 'd'), (0, -1, 'l'), (0, 1, 'r')]

    for dx, dy, move_dir in directions:
        new_row, new_col = x_row + dx, x_col + dy

        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_pos = new_row * 3 + new_col
            new_state = state[:]
            new_state[x_pos], new_state[new_pos] = new_state[new_pos], new_state[x_pos]
            neighbors.append((new_state, move_dir))

    return neighbors

def astar(initial_state):
    frontier = []
    heapq.heappush(frontier, (heuristic(initial_state), 0, [], initial_state)) 
    explored = set()

    while frontier:
        priority, cost, path, current_state = heapq.heappop(frontier)
    
        if is_target(current_state):
            return ''.join(reversed(path))

        explored.add(tuple(current_state))
          
        for neighbor_state, move_dir in get_neighbors(current_state, ''):
            if tuple(neighbor_state) not in explored:
                new_path = path + [move_dir]
                new_priority = heuristic(neighbor_state) + cost + 1
                heapq.heappush(frontier, (new_priority, cost + 1, new_path, neighbor_state))

    return 'unsolvable'

input_str = input().strip()
initial_state = [int(num) if num.isdigit() else 'x' for num in input_str.split()]
action_sequence = astar(initial_state)
print(action_sequence)
