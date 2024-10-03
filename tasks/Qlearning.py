import numpy as np
from utilities.Helpers import ActivationFunction


class Qlearning:
    def __init__(self, learning_rate: float = 0.5):
        self.learningRate = learning_rate

        self.sourceMatrix = None
        self.rewardMatrix = None
        self.agentMatrix = None

    def generate_matrices(self, environment: np.ndarray):
        self.sourceMatrix = environment

        rows = len(environment)     # 5
        cols = len(environment[0])  # 5

        # Init Agent matrix on zeros for each possible state -> Up, Down, Left, Right
        self.agentMatrix = np.zeros((rows*cols, rows*cols))

        # Init environment matrix on all -1
        self.rewardMatrix = np.full((rows**2, cols**2), -1)

        # Define moves (up, down, left, right)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Iterate over each cell in the environment matrix
        for r in range(rows):
            for c in range(cols):
                # Current cell in the environment matrix
                current_state = r * cols + c

                # Check if cell is wall or cheese
                if environment[r][c] == 1 or environment[r][c] == 3:
                    continue
                elif environment[r][c] == 4:
                    self.rewardMatrix[current_state][current_state] = 100

                # Check all possible moves from the current cell
                for move in moves:
                    # Calculate the next cell after making the move
                    next_r = r + move[0]
                    next_c = c + move[1]

                    # Check if the next cell is within the bounds of the environment matrix
                    if 0 <= next_r < rows and 0 <= next_c < cols:
                        next_state = next_r * rows + next_c

                        # Check if the next cell is not a wall
                        if environment[next_r][next_c] != 1:
                            # Update the reward for the next cell based on the current cell
                            if environment[next_r][next_c] == 3:  # Trap
                                self.rewardMatrix[current_state][next_state] = -100
                            elif environment[next_r][next_c] == 4:  # Cheese
                                self.rewardMatrix[current_state][next_state] = 100
                            else:  # Empty space or mouse
                                self.rewardMatrix[current_state][next_state] = 0

    def train(self, environment: np.ndarray, epochs: int = 25):
        # environment = np.array([[2, 1, 4], [0, 3, 0], [0, 0, 0]])
        self.generate_matrices(environment)

        # Define possible moves (up, down, left, right)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        possible_positions = []

        for i in range(len(environment)):
            for j in range(len(environment[i])):
                if environment[i][j] in [0, 2, 4]:  # Empty, mouse or cheese
                    possible_positions.append(i * len(environment) + j)

        # Theoretically while possiblePositions.len > 0
        for _ in range(epochs):
            # Take random possible place for the mouse and remove it from learning phase
            # state = np.random.choice(possible_positions, replace=True)
            if len(possible_positions) == 0:
                break

            state = possible_positions.pop()

            # Episode loop
            while True:
                # Select action randomly
                availableActions = np.where(self.rewardMatrix[state] > -1)[0]

                action = np.random.choice(availableActions)

                max_next_q_value = np.max(self.agentMatrix[action])
                self.agentMatrix[state][action] = self.rewardMatrix[state][action] + self.learningRate * max_next_q_value

                # Check if reached terminal state
                next_i, next_j = state // len(environment), state % len(environment)
                if environment[next_i][next_j] in [3, 4]:
                    break

                state = action

    def get_mouse(self, environment: np.ndarray) -> int:
        for i in range(len(environment)):
            for j in range(len(environment[i])):
                if environment[i][j] == 2:  # Empty, mouse or cheese
                    return i * len(environment) + j

    def find_path(self, environment: np.ndarray) -> list[int]:
        # environment = np.array([[2, 1, 4], [0, 3, 0], [0, 0, 0]])
        path = []
        current_state = self.get_mouse(environment)

        # Iterate until reaching a terminal state
        while True:
            # Add current state to the path
            path.append(current_state)

            # Get available actions based on Q-values
            available_actions = np.where(self.agentMatrix[current_state] != -1)[0]

            # Choose action with the highest Q-value
            action = np.argmax(self.agentMatrix[current_state][available_actions])

            # Update current state based on action
            next_state = available_actions[action]

            # Check if reached terminal state
            next_i, next_j = next_state // len(environment), next_state % len(environment)
            if environment[next_i][next_j] in [3, 4]:
                # Add terminal state to the path and break the loop
                path.append(next_state)
                break

            # Update current state
            current_state = next_state

        return path
