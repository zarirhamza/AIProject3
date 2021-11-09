from maze import Maze
from agent6 import Agent6
from agent7 import Agent7


def main():
    pathsMadeA6 = 0
    pathsMadeA7 = 0

    cellsProcessedA6 = 0
    cellsProcessedA7 = 0

    iterationsA6 = 0
    iterationsA7 = 0

    N = 10 #Times test is run
    Dimension = 25 #Dimension of the matrix
    TargetType = "random"# "random" // "forest" // "hill" // "flat"

    for i in range(N):
        m = Maze(Dimension, TargetType)  # Can choose target type as 'forest', 'hill', 'flat' as second parameter for testing
        # i. e Maze(10, 'forest')

        while not m.DFSSolve((0, 0), m.target):
            m = Maze(Dimension, TargetType)

        #m.plotMaze()

        a6 = Agent6(m)
        a7 = Agent7(m)

        iterations, cellsProcessed, stackPath = a6.run()
        iterationsA6 += iterations
        cellsProcessedA6 += cellsProcessed
        pathsMadeA6 += len(stackPath)

        iterations, cellsProcessed, stackPath = a7.run()
        iterationsA7 += iterations
        cellsProcessedA7 += cellsProcessed
        pathsMadeA7 += len(stackPath)

    print("Agent 6 iterations: {}".format(iterationsA6 / N))
    print("Agent 6 cellsProcessed: {}".format(cellsProcessedA6 / N))
    print("Agent 6 pathsTraveled: {}".format(pathsMadeA6 / N))

    print()

    print("Agent 7 iterations: {}".format(iterationsA7 / N))
    print("Agent 7 cellsProcessed: {}".format(cellsProcessedA7 / N))
    print("Agent 7 pathsTraveled: {}".format(pathsMadeA7 / N))


if __name__ == "__main__":
    main()
