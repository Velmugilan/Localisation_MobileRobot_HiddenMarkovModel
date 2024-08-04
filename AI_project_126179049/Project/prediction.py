import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors

    

def ReadFile():
    grid = []
    noisy_dist = []
    with open("./data.txt", "r") as fo:
        for i, line in enumerate(fo):
            if i >= 2 and i < 12:
                line_N = line.split()
                grid.append(line_N)

    data = pd.read_csv("robot_data.csv")
    t1 = data["Tower 1 Distance"].values
    t2 = data["Tower 2 Distance"].values
    t3 = data["Tower 3 Distance"].values
    t4 = data["Tower 4 Distance"].values

    grid = [[int(y) for y in x] for x in grid]
    grid = np.array(grid)
    for i in range(len(t1)):
        noisy_dist.append([t1[i], t2[i], t3[i], t4[i]])
    noisy_dist = np.array(noisy_dist)
    actual_path = list(zip(data["X"], data["Y"]))
    return grid, noisy_dist, actual_path



def Distance(t1, t2, Rx, Ry):
    x = t1
    y = t2
    d = math.sqrt(abs(Rx - x) ** 2 + abs(Ry - y) ** 2)
    return d



def HMM(grid, nd):
    n = np.count_nonzero(grid == 1) # counts the number of valid positions (non-obstacle cells) in the grid
    X = np.zeros((n, n)) # initializes the transition probability matrix X with zeros
    Z = np.zeros((10, 10)) # initializes the mapping matrix Z with zeros
    Z = Z.astype(int)
    p = 1
    for i in range(10): # check if a cell is a valid position (non-obstacle), assign integer
        for j in range(10):
            if (grid[i][j] != 0):
                Z[i][j] = p
                p = p + 1

    valid_positions = [] # Create a list of valid positions
    for i in range(10):
        for j in range(10):
            if grid[i][j] != 0:
                valid_positions.append((i, j))

    # Calculate transition probabilities based on valid positions (only cardinal directions)
    for i, (x1, y1) in enumerate(valid_positions):
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # Only consider cardinal directions
            x2, y2 = x1 + dx, y1 + dy
            if (x2, y2) in valid_positions:
                neighbors.append((x2, y2))

        if neighbors:
            for x2, y2 in neighbors:
                j = valid_positions.index((x2, y2))
                X[i][j] = 1 / len(neighbors)

    diagonal = 9 * math.sqrt(2)
    D1 = np.zeros(int(diagonal * 10))
    k = 0.0
    for i in range(int(diagonal * 10)):
        D1[i] = k
        k = k + 0.1

    E1 = Evidence(Z, D1, 0, 0, n, grid)
    E2 = Evidence(Z, D1, 0, 9, n, grid)
    E3 = Evidence(Z, D1, 9, 0, n, grid)
    E4 = Evidence(Z, D1, 9, 9, n, grid)
    '''
    X - Transition probability matrix
    Z - Mapping matrix (available states)
    E1, E2, E3, E4 - Emission matix (evidence)
    '''
    return X, Z, E1, E2, E3, E4, D1



def Evidence(Z, D1, t1, t2, n, grid):
    diagonal = 9 * math.sqrt(2) # maximum distance
    E = np.zeros((int(diagonal * 13), n))
    '''
    rows represent the possible distance measurements (with a range slightly larger than the diagonal distance)
    columns represent the valid positions (states) in the grid
    '''
    for i in range(10):
        for j in range(10):
            if Z[i][j] != 0 and grid[i][j] != 0:  # Check if the cell is a valid position
                d = round(Distance(t1, t2, i, j), 1)
                lb = round(d * 0.7, 1)
                ub = round(d * 1.3, 1) # assuming a 30% error range
                a = np.arange(lb, ub, 0.1)
                list_count = len(a)
                m = Z[i][j]
                for ind in range(int(diagonal * 10)):
                    if (np.isclose(lb, D1[ind])):
                        loc = ind
                        break
                for k in range(list_count):
                    E[loc + k][m - 1] = 1 / float(list_count) # sets the emission probability

    return E



def ForwardBackward(X, Z, E1, E2, E3, E4, nd, regularization_factor=1e-6):
    n = X.shape[0]
    T = len(nd)
    log_alpha = np.full((T, n), -np.inf)
    log_beta = np.full((T, n), -np.inf)
    gamma = np.zeros((T, n))
    epsilon = 1e-20  # Small positive constant to avoid log(0)

    for i in range(n):
        log_alpha[0][i] = np.log(1.0 / n) # log-probability of being in state i at time t, given the observations up to time t

    # Iterate through time steps for forward variable
    for t in range(1, T):
        for j in range(n):
            log_sum_val = -np.inf
            for i in range(n):
                log_transition = np.log(X[i][j] + regularization_factor)
                log_prob = log_alpha[t - 1][i] + log_transition
                log_sum_val = np.logaddexp(log_sum_val, log_prob)
                '''
                summing over the log-probabilities from the previous time step and the log-transition probabilities
                then adding the log-emission probability for the current observation
                '''
            e1 = max(E1[int(nd[t][0] * 10)][j], epsilon)
            e2 = max(E2[int(nd[t][1] * 10)][j], epsilon)
            e3 = max(E3[int(nd[t][2] * 10)][j], epsilon)
            e4 = max(E4[int(nd[t][3] * 10)][j], epsilon)
            log_emission = np.log(e1 * e2 * e3 * e4) # log of the product of the emission probabilities
            log_alpha[t][j] = log_sum_val + log_emission

    # Normalize log_alpha to get the probabilities
    for t in range(T):
        log_sum = np.logaddexp.reduce(log_alpha[t])
        log_alpha[t] -= log_sum

    # Compute gamma (probability of being in state i at time t)
    gamma = np.exp(log_alpha)

    return gamma



def Viterbi(X, Z, E1, E2, E3, E4, nd):
    n = X.shape[0]
    T = len(nd)
    delta = np.zeros((T, n))
    psi = np.zeros((T, n), dtype=int)
    epsilon = 1e-10  # Small positive constant to avoid divide-by-zero

    # Initialize base cases with prior knowledge about initial state probabilities
    start_state = 1
    for i in range(n):
        e1 = max(E1[int(nd[0][0] * 10)][i], epsilon)
        e2 = max(E2[int(nd[0][1] * 10)][i], epsilon)
        e3 = max(E3[int(nd[0][2] * 10)][i], epsilon)
        e4 = max(E4[int(nd[0][3] * 10)][i], epsilon)
        if i == start_state - 1:
            delta[0][i] = np.log(e1 * e2 * e3 * e4 + 1)  # Higher initial probability
        else:
            delta[0][i] = np.log(e1 * e2 * e3 * e4 + epsilon)  # Lower initial probability for other states

    # Iterate through time steps
    for t in range(1, T):
        for j in range(n):
            max_val = float('-inf')
            for i in range(n):
                if X[i][j] > 0:  # Check if transition is allowed
                    val = delta[t - 1][i] + np.log(X[i][j])
                    e1 = max(E1[int(nd[t][0] * 10)][j], epsilon)
                    e2 = max(E2[int(nd[t][1] * 10)][j], epsilon)
                    e3 = max(E3[int(nd[t][2] * 10)][j], epsilon)
                    e4 = max(E4[int(nd[t][3] * 10)][j], epsilon)
                    val += np.log(e1 * e2 * e3 * e4)
                    if val > max_val:
                        max_val = val
                        psi[t][j] = i + 1

            delta[t][j] = max_val

    # Find the most likely path
    path = [0] * T
    path[T - 1] = np.argmax(delta[T - 1]) + 1
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1][path[t + 1] - 1]

    return path



def Coord(location, Z):
    i, j = np.where(Z == location)
    return i[0], j[0]



def ViterbiAnimation(grid, path, Z):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

    cmap = plt.get_cmap('viridis')
    T = len(path)
    prob_map = np.zeros((10, 10))

    def init():
        grid_cmap = colors.ListedColormap(['black', 'white'])
        ax.imshow(grid, grid_cmap)
        return []

    def animate(t):
        prob_map.fill(0)
        x, y = Coord(path[t], Z)
        if grid[x][y] != 0:  # Check if the cell is not an obstacle
            prob_map[x][y] = 1.0

        im = ax.imshow(prob_map, cmap=cmap, vmin=0, vmax=1, alpha=0.5)
        return [im]

    ani = animation.FuncAnimation(fig, animate, frames=T, init_func=init, blit=True, repeat=False)
    plt.show()

    return ani



def DisplayPaths(grid, path, actual_path, Z):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

    cmap = plt.get_cmap('viridis')
    prob_map = np.zeros((10, 10))

    # Plot the grid
    grid_cmap = colors.ListedColormap(['black', 'white'])
    ax.imshow(grid, grid_cmap)

    # Plot the Viterbi path
    viterbi_x, viterbi_y = [], []
    for i, node in enumerate(path):
        x, y = Coord(node, Z)
        viterbi_x.append(x)
        viterbi_y.append(y)
    ax.plot(viterbi_y, viterbi_x, 'ro', markersize=8, label='Viterbi Path')

    # Plot the actual path
    actual_x, actual_y = [], []
    for t in range(len(actual_path)):
        x, y = actual_path[t]
        scaled_x = int((x - 25) / 50)
        scaled_y = int((y - 25) / 50)
        actual_x.append(scaled_x)
        actual_y.append(scaled_y)
    ax.plot(actual_x, actual_y, 'g--', label='Actual Path')

    ax.legend()
    plt.show()



grid, nd, actual_path = ReadFile()
print("Grid:")
print(grid)
X, Z, E1, E2, E3, E4, D1 = HMM(grid, nd)

path = Viterbi(X, Z, E1, E2, E3, E4, nd)
gamma = ForwardBackward(X, Z, E1, E2, E3, E4, nd)

print("\nProbable Path (Forward-Backward):")
for t in range(len(gamma)):
    max_idx = np.argmax(gamma[t])
    x, y = Coord(max_idx + 1, Z)
    print(f"Time Step {t}: ({x}, {y}) with probability {gamma[t][max_idx]:.4f}")

print("\nProbable Path (Viterbi):")
print("\nMost Likely Path:")
for i, node in enumerate(path):
    x, y = Coord(node, Z)
    print(f"Time Step {i}: ({x}, {y})")
    
ani_v = ViterbiAnimation(grid, path, Z)

DisplayPaths(grid, path, actual_path, Z)
