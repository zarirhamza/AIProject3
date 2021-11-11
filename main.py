import timeit
from maze import Maze
from agent6 import Agent6
from agent7 import Agent7
from agent8 import Agent8


def main():
    pathsMadeA6 = 0
    pathsMadeA7 = 0
    pathsMadeA8 = 0

    cellsProcessedA6 = 0
    cellsProcessedA7 = 0
    cellsProcessedA8 = 0

    iterationsA6 = 0
    iterationsA7 = 0
    iterationsA8 = 0

    cellsTraveledA6 = 0
    cellsTraveledA7 = 0
    cellsTraveledA8 = 0

    A6Time = 0
    A7Time = 0
    A8Time = 0

    N = 1 #Times test is run
    Dimension = 10 #Dimension of the matrix
    TargetType = "random"# "random" // "forest" // "hill" // "flat"

    for i in range(N):
        m = Maze(Dimension, TargetType)  # Can choose target type as 'forest', 'hill', 'flat' as second parameter for testing
        # i. e Maze(10, 'forest')

        while not m.DFSSolve((0, 0), m.target):
            m = Maze(Dimension, TargetType)

        m.plotMaze()

        a6 = Agent6(m)
        a7 = Agent7(m)
        a8 = Agent8(m)

        start = timeit.default_timer()
        iterations, cellsProcessed, stackPath = a6.run()
        stop = timeit.default_timer()
        A6Time += (stop - start)

        iterationsA6 += iterations
        cellsProcessedA6 += cellsProcessed
        pathsMadeA6 += len(stackPath)

        # Flatten list of paths to get all cells traveled
        cellsTraveledA6 += len([item for sublist in stackPath for item in sublist])

        #print(stackPath)

        start = timeit.default_timer()
        iterations, cellsProcessed, stackPath = a7.run()
        stop = timeit.default_timer()
        A7Time += (stop - start)

        iterationsA7 += iterations
        cellsProcessedA7 += cellsProcessed
        pathsMadeA7 += len(stackPath)

        # Flatten list of paths to get all cells traveled
        cellsTraveledA7 += len([item for sublist in stackPath for item in sublist])

        #print(stackPath)

        start = timeit.default_timer()
        iterations, cellsProcessed, stackPath = a8.run()
        stop = timeit.default_timer()
        A8Time += (stop - start)

        iterationsA8 += iterations
        cellsProcessedA8 += cellsProcessed
        pathsMadeA8 += len(stackPath)

        # Flatten list of paths to get all cells traveled
        cellsTraveledA8 += len([item for sublist in stackPath for item in sublist])

        # print(stackPath)

    print("Agent 6 iterations: {}".format(iterationsA6 / N))
    print("Agent 6 cellsProcessed: {}".format(cellsProcessedA6 / N))
    print("Agent 6 pathsTraveled: {}".format(pathsMadeA6 / N))
    print("Agent 6 cellsTraveled: {}".format(cellsTraveledA6 / N))
    print("Agent 6 Time: {}".format(A6Time / N))

    print()

    print("Agent 7 iterations: {}".format(iterationsA7 / N))
    print("Agent 7 cellsProcessed: {}".format(cellsProcessedA7 / N))
    print("Agent 7 pathsTraveled: {}".format(pathsMadeA7 / N))
    print("Agent 7 cellsTraveled: {}".format(cellsTraveledA7 / N))
    print("Agent 7 Time: {}".format(A7Time / N))

    print()

    print("Agent 8 iterations: {}".format(iterationsA8 / N))
    print("Agent 8 cellsProcessed: {}".format(cellsProcessedA8 / N))
    print("Agent 8 pathsTraveled: {}".format(pathsMadeA8 / N))
    print("Agent 8 cellsTraveled: {}".format(cellsTraveledA8 / N))
    print("Agent 8 Time: {}".format(A8Time / N))


if __name__ == "__main__":
    main()
