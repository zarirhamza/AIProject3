# Implementation of agent7

import random
import numpy as np
import heapq as hq

cellsProcessed = 0
iterations = 0


class Agent8:
    """
    init - Initialize agent based on requirements
    :param maze: Maze object to be inspected by agent
    """

    def __init__(self, maze):
        self.fullMaze = maze.maze
        self.dim = maze.dim
        self.numUnblocked = self.dim ** 2
        self.blankMaze = np.zeros([self.dim, self.dim], dtype=object)
        self.probMatrix = np.full((self.dim, self.dim), 1 / (self.dim ** 2))
        self.maxProb = 0

        self.targetCell = maze.target

        self.FNRMapping = {
            'L': 0.2,
            'H': 0.5,
            'F': 0.8,
            'X': 0,
            0: 0
        }

        global iterations
        iterations = 0

        global cellsProcessed
        cellsProcessed = 0

    """
    mDistance - Calculates Manhattan Distance between two points
    """

    def mDistance(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    """
    bestCell - Calculates best cell based on distance from currentCell
    :param currCell: Current cell being measured from
    """

    def bestCell(self, currCell):
        maxVal_x = np.unravel_index(np.argmax(self.probMatrix, axis=None), self.probMatrix.shape)[1]
        maxVal_y = np.unravel_index(np.argmax(self.probMatrix, axis=None), self.probMatrix.shape)[0]

        maxValues = [(maxVal_x, maxVal_y)]
        for i in range(self.dim):
            for j in range(self.dim):
                if self.probMatrix[j][i] == self.probMatrix[maxVal_y][maxVal_x]:
                    if self.mDistance(currCell[0], currCell[1], i, j) < self.mDistance(currCell[0], currCell[1],
                                                                                       maxVal_x, maxVal_y):
                        maxVal_x = i
                        maxVal_y = j
                        maxValues = [(maxVal_x, maxVal_y)]
                    elif self.mDistance(currCell[0], currCell[1], i, j) == self.mDistance(currCell[0], currCell[1],
                                                                                          maxVal_x, maxVal_y):
                        maxValues.append((i, j))

        randVal = random.randint(1, len(maxValues)) - 1
        return maxValues[randVal]

    """
    checkMaxChange - Checks if max has changed
    :param currMax: Current max probability to be compared to
    """

    def checkMaxChanged(self, currMax):

        maxVal_x = np.unravel_index(np.argmax(self.probMatrix, axis=None), self.probMatrix.shape)[1]
        maxVal_y = np.unravel_index(np.argmax(self.probMatrix, axis=None), self.probMatrix.shape)[0]
        maxVal = self.probMatrix[maxVal_y][maxVal_x]

        if self.probMatrix[currMax[1]][currMax[0]] < maxVal:
            return True
        else:
            return False

    """
    run - Runs agent to find target
    """

    def run(self):

        stackPath = []
        currentCell = (0, 0)
        start = (0, currentCell[0], currentCell[1])
        fringe = [start]
        hq.heapify(fringe)
        closedList = []
        gn = [[(np.inf, None) for i in range(self.dim)] for j in range(self.dim)]
        gn[currentCell[1]][currentCell[0]] = (0, None)
        global iterations

        while True:

            # get Max Cell
            tempTarget = self.bestCell(currentCell)

            # A* to max Cell
            self.helperAStar(self.blankMaze, tempTarget, self.mDistanceHeurisitic, fringe, gn, closedList)
            path = self.createPath(currentCell, tempTarget, gn)

            # Iterate through planned path while updating values
            discovered, stopPoint = self.checkPath(path)
            stackPath.append(path)

            # End if target is found
            if discovered and stopPoint == self.targetCell:
                global cellsProcessed
                return iterations, cellsProcessed, stackPath

            # If no path found, mark off target as unreachable (blocked) and update probabilities
            elif not discovered and stopPoint == (-1, -1):
                iterations += 1

                #print(self.fullMaze)
                #print(self.blankMaze)
                #print(self.numUnblocked)
                #print(self.probMatrix)
                #print(closedList)
                #print(len(closedList))

                #mark everything not in closed list as impossible to reach
                for i in range(self.dim):
                    for j in range(self.dim):
                        if len([item for item in closedList if item[1] == i and item[2] == j]) == 0:

                            #print("i: {} and j: {} where closedList is there".format(i,j))
                            if self.probMatrix[j][i] != 0:
                                self.probMatrix[j][i] = 0
                                self.probMatrix = np.asarray([
                                    [val if val == 0 else val + (1 / (self.numUnblocked * (self.numUnblocked - 1))) for val in row] for
                                    row in self.probMatrix], dtype=object)
                                self.numUnblocked -= 1

                start = (0, currentCell[0], currentCell[1])
                fringe = [start]
                hq.heapify(fringe)
                closedList = []
                gn = [[(np.inf, None) for i in range(self.dim)] for j in range(self.dim)]
                gn[currentCell[1]][currentCell[0]] = (0, (gn[currentCell[1]][currentCell[0]][1]))

            # Otherwise restart A* with new start as end point
            else:
                currentCell = stopPoint
                start = (0, stopPoint[0], stopPoint[1])
                fringe = [start]
                hq.heapify(fringe)
                closedList = []
                gn = [[(np.inf, None) for i in range(self.dim)] for j in range(self.dim)]
                gn[stopPoint[1]][stopPoint[0]] = (0, (gn[stopPoint[1]][stopPoint[0]][1]))

    """
    createChildren - Generates children of given point and stores in (fn,x,y) form in fringe when needed 
    :param x: Point to generate children
    :param maze: Grid to solve with blocked/unblocked cells
    :param G: Goal Point
    :param heuristic: Heuristic calculation method
    :param fringe: Priority Queue of different cells
    :param gn: Matrix containing distances of all cells and parents
    :param closedList: List containg visited cells
    """
    def createChildren(self, x, maze, G, heuristic, fringe, gn, closedList):
        xc = x[1]
        yc = x[2]

        #print("gn is {}".format(gn))
        #print("closedList is {}".format(closedList))

        # generate legal children for all 4 possible moves based on origin point
        for direction in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            newxc = direction[0] + xc
            newyc = direction[1] + yc

            newgn = gn[yc][xc][0] + 1
            newfn = heuristic((newxc, newyc), G) + newgn

            #print("Children is: {} with gn of {} and fn of {}".format((newxc, newyc), newgn, newfn))
            #input()

            if newxc >= self.dim or newxc < 0 or newyc >= self.dim or newyc < 0:  # Legal within bounds children only!
                continue

            elif maze[newyc][newxc] == 'X':  # add only if open
                continue

            elif len([item for item in closedList if item[1] == newxc and item[2] == newyc and item[0] <= newfn]) > 0:
                continue

            elif len([item for item in fringe if item[1] == newxc and item[2] == newyc and item[0] <= newfn]) > 0:
                continue

            gn[newyc][newxc] = (newgn, (xc, yc))
            item = (newfn, newxc, newyc)
            hq.heappush(fringe, item)

            if newxc == G[0] and newyc == G[1]:
                #print("Goal found")
                return

        return

    """
    helperAStar - A* Child Generation and Path Checking
    :param maze: Grid to solve with blocked/unblocked cells
    :param G: Goal Point
    :param heuristic: Heuristic calculation method
    :param fringe: Priority Queue of different cells
    :param gn: Matrix containing distances of all cells and parents
    :param closedList: List containg visited cells
    """
    def helperAStar(self, maze, G, heuristic, fringe, gn, closedList):
        if not fringe:  # check for empty fringe meaning no remaining paths to check
            return

        while fringe:
            #print("fringe is {}".format(fringe))

            global cellsProcessed
            cellsProcessed += 1
            x = hq.heappop(fringe)  # pop first element and check if goal otherwise create children and call again
            closedList.append(x)
            #print("lowest fn of fringe is {}".format(x))

            if x[1] == G[0] and x[2] == G[1]:
                #print(gn)
                return
            self.createChildren(x, maze, G, heuristic, fringe, gn, closedList)
        return

    """
    checkPath - Path traversal and updating method
    :param path: Proposed path to iterate through
    """

    def checkPath(self, path):

        # No path passed in means target is unreachable
        if not path:
            return False, (-1, -1)

        # Iterate through path adding information when necessary
        for i in range(len(path)):

            global iterations
            newCell = False

            # Update knowledge if cell is unvisited
            if self.blankMaze[path[i][1]][path[i][0]] == 0:
                self.blankMaze[path[i][1]][path[i][0]] = self.fullMaze[path[i][1]][path[i][0]]
                newCell = True

            # Check if cell being moved into is the target cell
            if (i > 0 or i == len(path) - 1) and path[i] == self.targetCell:
                terrainFNR = self.FNRMapping[self.blankMaze[path[i][1]][path[i][0]]]
                p = random.uniform(0, 1)
                # print("Terrain FNR: ", terrainFNR, " Probability: ", p)
                if p > terrainFNR:
                    iterations += 1
                    self.probMatrix = np.zeros([self.dim, self.dim], dtype=object)
                    self.probMatrix[path[i][1]][path[i][0]] = 1
                    return True, path[i]

            # If blockage found, update knowledge and stop path
            if newCell and self.blankMaze[path[i][1]][path[i][0]] == 'X':
                iterations += 1
                self.probMatrix[path[i][1]][path[i][0]] = 0
                self.probMatrix = np.asarray([
                    [val if val == 0 else val + (1 / (self.numUnblocked * (self.numUnblocked - 1))) for val in row] for
                    row in self.probMatrix], dtype=object)
                self.numUnblocked -= 1

                # Update probability as per A7 rules via Q3
                self.probMatrix = np.asarray([
                    [val if val == 0 else val * (1 - (self.FNRMapping[self.blankMaze[path[i][1]][path[i][0]]])) for val
                     in row] for
                    row in self.probMatrix], dtype=object)

                return False, path[i - 1]

            # Otherwise move into new cell in path and update probability based on Q2
            elif i > 0 or i == len(path) - 1:
                iterations += 1
                self.probMatrix[path[i][1]][path[i][0]] *= self.FNRMapping[self.blankMaze[path[i][1]][path[i][0]]]
                probSum = np.sum(self.probMatrix)
                self.probMatrix = self.probMatrix / probSum

                # Update probability as per A7 rules via Q3
                self.probMatrix = np.asarray([
                    [val if val == 0 else val * (1 - (self.FNRMapping[self.blankMaze[path[i][1]][path[i][0]]])) for val
                     in row] for row in self.probMatrix], dtype=object)

                # Check if new maximum probability and return end of path if so
                if self.checkMaxChanged(path[-1]):
                    return False, path[i]

        # Goal not found and return last point to start A* from
        return False, path[-1]

    """
    createPath - Create path given the matrix with reversible path from G
    :param maze: Grid to solve with blocked/unblocked cells
    :param S: Start Point
    :param G: Goal Point
    :param gn: Matrix containing distances of all cells and parents
    :return: the path given as a list
    """

    def createPath(self, S, G, gn):
        # two arrays, one to store the directions in reverse, and one to give the actual directions
        path = [(G[0], G[1])]

        # stores the current coordinates
        xCoord = G[0]
        yCoord = G[1]

        # stores the coordinates of start
        xStart = S[0]
        yStart = S[1]

        # check if goal is reachable
        if gn[G[1]][G[0]][0] == np.inf:
            return []

        # assemble coordinates of path from goal to start before reversing it
        while not (xCoord == xStart and yCoord == yStart):
            # Retrieves parents coords
            nextMove = gn[yCoord][xCoord][1]
            # print(nextMove)

            if nextMove is None:
                break

            # calculates the next coordinate
            xCoord = nextMove[0]
            yCoord = nextMove[1]

            # append to path
            path.append((xCoord, yCoord))

        return list(reversed(path))

    """
    mDistance - calculates manhattan distance from goal
    :param x: point whose priority will be evaluated
    :param G: goal
    :return: priority number based on speed
    """
    def mDistanceHeurisitic(self, x, G):
        xCoord = x[0]
        yCoord = x[1]
        xdist = G[0] - xCoord
        ydist = G[1] - yCoord
        h = abs(xdist) + abs(ydist)
        return h