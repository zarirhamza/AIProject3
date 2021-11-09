from maze import Maze
from agent6 import Agent6
from agent7 import Agent7


def main():
    pathsMadeA6 = 0
    pathsMadeA7 = 0

    cellsTraveledA6 = 0
    cellsTraveledA7 = 0

    cellsProcessedA6 = 0
    cellsProcessedA7 = 0

    iterationsA6 = 0
    iterationsA7 = 0

    N = 100

    for i in range(N):
        m = Maze(10)  # Can choose target type as 'forest', 'hill', 'flat' as second parameter for testing
        # i. e Maze(10, 'forest')

        while not m.DFSSolve((0, 0), m.target):
            m = Maze(10)

        a6 = Agent6(m)
        a7 = Agent7(m)

        iterations, cellsProcessed, stackPath = a6.run()
        iterationsA6 += iterations
        cellsProcessedA6 += cellsProcessed
        pathsMadeA6 += len(stackPath)

        # Flatten list of paths to get all cells traveled
        cellsTraveledA6 += len([item for sublist in stackPath for item in sublist])

        iterations, cellsProcessed, stackPath = a7.run()
        iterationsA7 += iterations
        cellsProcessedA7 += cellsProcessed
        pathsMadeA7 += len(stackPath)

        # Flatten list of paths to get all cells traveled
        cellsTraveledA7 += len([item for sublist in stackPath for item in sublist])

    print("Agent 6 iterations: {}".format(iterationsA6 / N))
    print("Agent 6 cellsProcessed: {}".format(cellsProcessedA6 / N))
    print("Agent 6 pathsTraveled: {}".format(pathsMadeA6 / N))
    print("Agent 6 cellsTraveled: {}".format(cellsTraveledA6 / N))

    print()

    print("Agent 7 iterations: {}".format(iterationsA7 / N))
    print("Agent 7 cellsProcessed: {}".format(cellsProcessedA7 / N))
    print("Agent 7 pathsTraveled: {}".format(pathsMadeA7 / N))
    print("Agent 7 cellsTraveled: {}".format(cellsTraveledA7 / N))


if __name__ == "__main__":
    main()
