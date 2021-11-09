# Maze class to generate grid and relevant helper functions
import random

import numpy as np


class Maze:
    """
    init - Create maze with proper probabilities and terrain diversity
    :param dim: Dimensions
    :param target_type: Desired target terrain type
    """

    def __init__(self, dim, targetType=None):
        self.dim = dim
        self.target = ()
        self.maze = self.createMaze()
        #self.maze = np.array([['L', 'L', 'F'], ['X', 'F', 'H'], ['X', 'F', 'H']])

        # Make the target
        self.targetType = targetType
        self.target = self.createTarget()
        #self.target = (1, 2)

    """
    createMaze - Fill out map matrix with values representing terrain and target
    """

    def createMaze(self):
        mat = np.zeros([self.dim, self.dim], dtype=object)
        for i in range(self.dim):
            for j in range(self.dim):
                p = np.random.rand()
                if p <= 0.3:
                    mat[i][j] = 'X'  # Blocked
                else:
                    terrain = random.uniform(0, 1)# Change probabilities to affect terrain generation
                    if terrain <= 0.33:
                        mat[i][j] = "L"  # Flat
                    elif 0.33 < terrain <= 0.67:
                        mat[i][j] = "H"  # Hilly
                    elif terrain > 0.67: 
                        mat[i][j] = "F"  # Forest

        terrain = np.random.randint(3)
        if terrain == 0:
            mat[0][0] = "L"  # Flat
        elif terrain == 1:
            mat[0][0] = "H"  # Hilly
        elif terrain == 2:
            mat[0][0] = "F"  # Forest

        return mat

    def createTarget(self):
        while True:
            x = np.random.randint(self.dim)
            y = np.random.randint(self.dim)

            if self.targetType is None or self.targetType == "random":
                if self.maze[y][x] != 'X':
                    return x, y

            elif self.targetType == "flat":
                if self.maze[y][x] == 'L':
                    return x, y

            elif self.targetType == "hill":
                if self.maze[y][x] == 'H':
                    return x, y

            elif self.targetType == "forest":
                if self.maze[y][x] == 'F':
                    return x, y

    """
    plotMaze - Prints maze in terminal
    """

    def plotMaze(self):
        for x in range(self.dim):
            line = ""
            for y in range(self.dim):
                if x != self.target[1] or y != self.target[0]:
                    line = line + " " + str(self.maze[x][y]) + " "
                else:
                    line = line + " " + str(self.maze[x][y]) + "!"
            print(line)
        print()

    """
    DFSSolve - Checks if G is reachable from S using DFS
    :param S: tuple representing (S)tarting square
    :param G: tuple representing (G)oal square
    :return: boolean stating whether G is reachable from S
    """

    def DFSSolve(self, S, G):
        visited = []
        stack = [S]

        while len(stack):
            v = stack.pop()
            if v not in visited:
                visited.append(v)
                if v == G:
                    return True
                stack.extend(self.findNeighbors(v))
        return False

    """
    findNeighbors - Determines all valid neighbors for passed in Loc
    :param loc: Location to find neighbors of
    :return: List of valid neighbors
    """

    def findNeighbors(self, loc):
        neighbors = []
        for move in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            neighbor = (loc[0] + move[0], loc[1] + move[1])
            if 0 <= neighbor[0] < self.dim and 0 <= neighbor[1] < self.dim and self.maze[neighbor[1]][
                neighbor[0]] != 'X':
                neighbors.append(neighbor)
        return neighbors
