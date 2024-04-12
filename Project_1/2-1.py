from collections import deque

def is_target(state):
    return state == [1, 2, 3, 4, 5, 6, 7, 8, 'x']

def get_neighbors(state):
    neighbors = []
    x_pos = state.index('x')
    x_row, x_col = divmod(x_pos, 3)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dx, dy in directions:
        new_row, new_col = x_row + dx, x_col + dy

        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_pos = new_row * 3 + new_col
            new_state = state[:]
            new_state[x_pos], new_state[new_pos] = new_state[new_pos], new_state[x_pos]

            neighbors.append(new_state)

    return neighbors

def solve_puzzle(initial_state):
    queue = deque([(initial_state, 0)])
    visited = set([tuple(initial_state)])

    while queue:
        current_state, steps = queue.popleft()

        if is_target(current_state):
            return steps
        for neighbor in get_neighbors(current_state):
            if tuple(neighbor) not in visited:
                queue.append((neighbor, steps + 1))
                visited.add(tuple(neighbor))

    return -1
input_str = input().strip()
initial_state = [int(num) if num.isdigit() else 'x' for num in input_str.split()]

result = solve_puzzle(initial_state)
if result != -1:
    print(result)
else:
    print(0)
