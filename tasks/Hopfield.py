import numpy as np
from utilities.Helpers import ActivationFunction, activate_function_x


class Hopfield:
    def __init__(self, use_async: bool, function: ActivationFunction = ActivationFunction.Signum):
        self.useAsync = use_async
        self.function = function

        self.patterns = []
        self.shape = None
        self.savedPatterns = None

    # Requires Matrix input
    def train(self, pattern: np.ndarray):
        self.patterns.append(pattern)
        self.shape = len(pattern)

        if self.savedPatterns is None:
            # Generate empty matrix for pattern saving
            self.savedPatterns = np.zeros((self.shape**2, self.shape**2))

        # pattern => patternMatrix
        # [[1, 2], [3, 4]] => [r1, r2] => [r1 + r2] => [1, 2, 3, 4]
        patternVector = pattern.flatten()
        # Replace 0 -> -1
        patternVector = np.where(patternVector == 0, -1, patternVector)
        # Weight matrix for a pattern
        weight_pattern = np.outer(patternVector, patternVector)

        self.savedPatterns += weight_pattern

        np.fill_diagonal(self.savedPatterns, 0)

    # Requires Matrix input
    def train_legacy(self, patterns: list[np.ndarray]):
        self.patterns = patterns
        self.shape = len(patterns[0])

        if self.savedPatterns is None:
            # Generate empty matrix for pattern saving
            self.savedPatterns = np.zeros((self.shape ** 2, self.shape ** 2))

        # pattern => patternMatrix
        for pattern in patterns:
            # [[1, 2], [3, 4]] => [r1, r2] => [r1 + r2] => [1, 2, 3, 4]
            patternVector = pattern.flatten()
            # Replace 0 -> -1
            patternVector = np.where(patternVector == 0, -1, patternVector)
            # Weight matrix for a pattern
            weight_pattern = np.outer(patternVector, patternVector)

            self.savedPatterns += weight_pattern

        np.fill_diagonal(self.savedPatterns, 0)

    # Requires Matrix input
    def get_result(self, pattern: np.ndarray, epochs: int = 1) -> np.ndarray:
        patternVector = pattern.flatten()

        if self.useAsync:
            return self.async_result(epochs, patternVector)

        return self.sync_result(epochs, patternVector)

    def async_result(self, epochs: int, pattern: np.ndarray) -> np.ndarray:
        # Row based format to column based one
        weights = self.savedPatterns.transpose()
        # 0 => -1
        cleanPattern = np.where(pattern == 0, -1, pattern)

        if len(weights) != len(cleanPattern):
            raise Exception("Missmach in operation size if asynchronous recovery of pattern")

        # Set maximum iterations ?
        for _ in range(epochs):
            # Number of rows = cols => iterate and update result by it
            for i, weight_i in enumerate(weights):
                dotResult = float(np.dot(cleanPattern, weight_i))
                # There should not be a situation where the result of sign will be = 0
                # V_i = sign(dot(V*W_i))
                cleanPattern[i] = activate_function_x(dotResult, self.function)

        cleanPattern = np.where(cleanPattern == -1, 0, cleanPattern)

        return cleanPattern.reshape(self.shape, -1)

    def sync_result(self, epochs: int, pattern: np.ndarray) -> np.ndarray:
        cleanPattern = np.where(pattern == 0, -1, pattern)

        # Set maximum iterations ?
        for _ in range(epochs):
            tmpPattern = np.dot(self.savedPatterns, cleanPattern)
            tmpPattern = np.sign(tmpPattern)
            cleanPattern = tmpPattern

        cleanPattern = np.where(pattern == -1, 0, pattern)

        return cleanPattern.reshape(self.shape, -1)
