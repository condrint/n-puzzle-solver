import random
import copy 
import time
from heapq import heappush, heappop

"""
IMPORTANT: float('inf') = infinity => represents the empty square in this implementation
"""

class Board:
    def __init__ (self, startingBoard=None, goalState=None, n=3, disableInvalidPuzzles=True):
        """
        initialize an n * n matrix represented by a 1d list
        contains an empty spot and integers 1, 2.... n*n - 1
        """
        self.limit = 2 ** 200
        self.n = n
        self.goalState = goalState or [i + 1 for i in range(self.n ** 2 - 1)] + [float('inf')] # n=3 will initialize as [1, 2, 3, 4, 5, 6, 7, 8, None]

        if startingBoard and isinstance(startingBoard, list) and len(startingBoard) == 9:
            self.state = startingBoard
        else:
            self.state = copy.deepcopy(self.goalState)
            if disableInvalidPuzzles:
                random.shuffle(self.state)
                while self.isUnsolvableState(self.state):
                    random.shuffle(self.state)
            else:
                random.shuffle(self.state)
            
    
    def isUnsolvableState(self, state):
        # inspired by https://www.geeksforgeeks.org/check-instance-8-puzzle-solvable/
        inversions = 0
        for i in range(len(state) - 1):
            if state[i] == float('inf'):
                continue
            for j in range(i + 1, len(state)):
                if state[j] == float('inf'):
                    continue
                if state[i] > state[j]: 
                    inversions += 1
        return inversions % 2 == 1

    def printBoard(self):
        for i in range(1, self.n + 1):
            print(self.state[self.n * (i - 1):self.n * i])


    def solveWithBFS(self, likeSlide48Graphic=False):
        """
        solves current board state
        with breadth first search
        """

        # initialize queue with current state
        queue = [self.state]
        seenStates = set()
        seenStates.add(tuple(self.state)) # lists are unhashable so the state must be converted to a tuple and checked as a tuple
        depth = 0 
        steps = 0
        while queue:
            nextLevelOfQueue = []
            while queue:
                if likeSlide48Graphic:
                    # pops from the front to maintain a consistent order
                    # WAAAY slower than .pop(), runs in O(q) time where
                    # q is the longest length of the queue 
                    # which I think is factorial(self.n) / 2
                    state = queue.pop(0)
                else:
                    # .pop() runs in O(1) time
                    state = queue.pop()
                steps += 1

                if state == self.goalState:
                    print('Solved puzzle with depth of ' + str(depth) + ' in ' + str(steps) + ' steps.')
                    return True

                # make a deep copy to avoid unexpectedly 
                # copying a pointer and swapping multiple 
                # states in the queue
                indexOfEmptySquare = state.index(float('inf'))
                iOfE = indexOfEmptySquare

                # swap in the following order: 
                # left, up, right, down                    LEFT        UP           RIGHT       DOWN            bounds checking
                for indexToSwapWithEmpty in [i for i in [iOfE - 1, iOfE - self.n, iOfE + 1, iOfE + self.n] if 0 <= i < self.n ** 2 # these next 2 lines prevent cheating 2d array by using 1 dimension
                                                                                                            and not (iOfE % self.n == 0 and i == iOfE - 1) 
                                                                                                            and not ((iOfE + 1) % self.n == 0 and i == iOfE + 1)]:
                    newState = copy.deepcopy(state)
                    newState[indexOfEmptySquare], newState[indexToSwapWithEmpty] = newState[indexToSwapWithEmpty], newState[indexOfEmptySquare]
                    if tuple(newState) in seenStates:
                        continue
                    
                    seenStates.add(tuple(newState))
                    nextLevelOfQueue.append(newState)

            queue = nextLevelOfQueue
            depth += 1

        # algorithm will get here for boards of odd number of inflections
        print('Could not solve puzzle.')
        return False

    
    def solveWithHamming(self):
        """
        solves current board state
        with A* search using
        hamming distance as a heuristic

        uses python's min heap to store
        each state and it's hamming value of
        "number of blocks in the wrong position, 
        plus the number of moves made so far to get to the state"
        """

        # initialize queue with current state
        queue = []
        distance = self.hammingDistance(self.state)
        heappush(queue, (distance + 0, distance, self.state)) #heap(hammingdistance + steps, hammingdistance to subtract from total, state)

        seenStates = set()
        seenStates.add(tuple(self.state)) # lists are unhashable so the state must be converted to a tuple and checked as a tuple
        steps = 0

        while queue:
            currentNode = heappop(queue)
            state, currentStateSteps = currentNode[2], currentNode[0] - currentNode[1]
            steps += 1
            if state == self.goalState:
                print('Solved puzzle in ' + str(steps) + ' steps.')
                return True

            # make a deep copy to avoid unexpectedly 
            # copying a pointer and swapping multiple 
            # states in the queue
            indexOfEmptySquare = state.index(float('inf'))
            iOfE = indexOfEmptySquare

            # swap in the following order: 
            # left, up, right, down                    LEFT        UP           RIGHT       DOWN            bounds checking
            for indexToSwapWithEmpty in [i for i in [iOfE - 1, iOfE - self.n, iOfE + 1, iOfE + self.n] if 0 <= i < self.n ** 2 # these next 2 lines prevent traversing cheating 2d array
                                                                                                        and not (iOfE % self.n == 0 and i == iOfE - 1) 
                                                                                                        and not ((iOfE + 1) % self.n == 0 and i == iOfE + 1)]:
                newState = copy.deepcopy(state)
                newState[indexOfEmptySquare], newState[indexToSwapWithEmpty] = newState[indexToSwapWithEmpty], newState[indexOfEmptySquare]
                if tuple(newState) in seenStates:
                    continue
                
                seenStates.add(tuple(newState))
                distance = self.hammingDistance(newState)
                heappush(queue, (currentStateSteps + 1 + distance, distance, newState))

        # algorithm will get here for boards of odd number of inflections
        print('Could not solve puzzle.')
        return False

    
    def hammingDistance(self, state):
        """
        returns hamming distance of state
        and goal state
        
        runs in O(self.n) time, but usually n is very small.
        If it gets very large, this function is the least of 
        your computer's concerns 
        """
        return sum([1 if state[i] != self.goalState[i] else 0 for i in range(len(state))])


    def solveWithManhattan(self):
        """
        solves current board state
        with A* search using
        manhattan distance as a heuristic

        uses python's min heap to store
        each state and it's manhattan value of
        "The sum of the distances (sum of the vertical and horizontal distance) from the blocks
        to their goal positions, plus the number of moves made so far to get to the state."
        """

        # initialize queue with current state
        queue = []
        distance = self.manhattanDistance(self.state)
        heappush(queue, (distance + 0, distance, self.state)) #heap(manhattandistance + steps, mannhattandistance to subtract from total, state)

        seenStates = set()
        seenStates.add(tuple(self.state)) # lists are unhashable so the state must be converted to a tuple and checked as a tuple
        steps = 0

        while queue:
            currentNode = heappop(queue)
            #print(currentNode)
            state, currentStateSteps = currentNode[2], currentNode[0] - currentNode[1]
            steps += 1
            if state == self.goalState:
                print('Solved puzzle with in ' + str(steps) + ' steps.')
                return True

            # make a deep copy to avoid unexpectedly 
            # copying a pointer and swapping multiple 
            # states in the queue
            indexOfEmptySquare = state.index(float('inf'))
            iOfE = indexOfEmptySquare

            # swap in the following order: 
            # left, up, right, down                    LEFT        UP           RIGHT       DOWN            bounds checking
            for indexToSwapWithEmpty in [i for i in [iOfE - 1, iOfE - self.n, iOfE + 1, iOfE + self.n] if 0 <= i < self.n ** 2 # these next 2 lines prevent traversing cheating 2d array
                                                                                                        and not (iOfE % self.n == 0 and i == iOfE - 1) 
                                                                                                        and not ((iOfE + 1) % self.n == 0 and i == iOfE + 1)]:
                newState = copy.deepcopy(state)
                newState[indexOfEmptySquare], newState[indexToSwapWithEmpty] = newState[indexToSwapWithEmpty], newState[indexOfEmptySquare]
                if tuple(newState) in seenStates:
                    continue
                
                seenStates.add(tuple(newState))
                distance = self.manhattanDistance(newState)
                heappush(queue, (currentStateSteps + 1 + distance, distance, newState))


        # algorithm will get here for boards of odd number of inflections
        print('Could not solve puzzle.')
        return False

    
    def manhattanDistance(self, state):
        """
        returns hamming distance of state
        and goal state
        
        runs in O(self.n) time, but usually n is very small.
        If it gets very large, this function is the least of 
        your computer's concerns 
        """
        manhattan = 0
        for i, value in enumerate(state):
            value -= 1 #account for index by zero 
            if value == float('inf'):
                continue
            verticalDistance = abs(i // self.n - value // self.n)
            horizontalDistance = abs(i % self.n - value % self.n)
            manhattan += verticalDistance + horizontalDistance
        return manhattan
            


            
if __name__ == "__main__":
    board = Board([8, 1, 3, 4, float('inf'), 2, 7, 6, 5], None, 3)
    board.printBoard()
    
    print('\nBFS:')
    start = time.time()
    board.solveWithBFS()
    end = time.time()
    print('Ran in ' + str(end - start) + ' seconds.')

    print('\nA* search with Hamming distance as heuristic:')
    start = time.time()
    board.solveWithHamming()
    end = time.time()
    print('Ran in ' + str(end - start) + ' seconds.')
    
    print('\nA* search with Manhattan distance as heuristic:')
    start = time.time()
    board.solveWithManhattan()
    end = time.time()
    print('Ran in ' + str(end - start) + ' seconds.')