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
        self.n = n
        self.goalState = goalState or [i + 1 for i in range(self.n ** 2 - 1)] + [float('inf')] # n=3 will initialize as [1, 2, 3, 4, 5, 6, 7, 8, None]

        if startingBoard and isinstance(startingBoard, list) and len(startingBoard) == n*n:
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
        pathQueue = [[self.state]]
        seenStates = set()
        seenStates.add(tuple(self.state)) # lists are unhashable so the state must be converted to a tuple and checked as a tuple
        depth = 0 
        steps = 0

        while queue:

            nextLevelOfQueue = []
            nextLevelOfPathQueue = []
            
            while queue:
                if likeSlide48Graphic:
                    # pops from the front to maintain a consistent order
                    # WAAAY slower than .pop(), runs in O(q) time where
                    # q is the longest length of the queue 
                    # which I think is factorial(self.n ** 2) / 2
                    state = queue.pop(0)
                    path = pathQueue.pop(0)
                else:
                    # .pop() runs in O(1) time
                    state = queue.pop()
                    path = pathQueue.pop()
                steps += 1

                if state == self.goalState:
                    print('Solved puzzle with depth of ' + str(depth) + ' in ' + str(steps) + ' steps.')
                    return path

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
                    
                    newPath = copy.deepcopy(path)
                    newPath.append(newState)
                    seenStates.add(tuple(newState))
                    nextLevelOfQueue.append(newState)
                    nextLevelOfPathQueue.append(newPath)

            queue = nextLevelOfQueue
            pathQueue = nextLevelOfPathQueue 
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
        returns manhattan distance of state
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

    
    def BFS_likeSlide(self, likeSlide48Graphic=True):
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
       edges = [] # helps generate tree
       nodes = {} # helps generate tree

       while queue:
           nextLevelOfQueue = []

           while queue:
               if likeSlide48Graphic:
                   # pops from the front to maintain a consistent order
                   # WAAAY slower than .pop(), runs in O(q) time where
                   # q is the longest length of the queue 
                   # which I think is factorial(self.n ** 2) / 2
                   state = queue.pop(0)
               else:
                   # .pop() runs in O(1) time
                   state = queue.pop()

               steps += 1
               if steps > 46:
                   self.displayTree(edges, nodes)
                   return

               nodes[steps] = self.generateString(state)
                
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
                   if steps + len(nextLevelOfQueue) + len(queue) <= 46:
                        edges.append([steps, steps + len(nextLevelOfQueue) + len(queue)]) # tricky math to predict steps needed to get to newly added state 

           queue = nextLevelOfQueue
           depth += 1
       # algorithm will get here for boards of odd number of inflections
       print('Could not solve puzzle.')
       return False
    

    def generateString(self, state):
        """
        returns string representing state
        """
        string = ''
        for i, val in enumerate(state):
            if i != 0 and i % self.n == 0:
                string += '\n'
            if val == float('inf'):
                string += '[]'
            else:
                string += str(val)
            if (i + 1) % self.n != 0 and i + 1 < len(state):
                string += ' '
        return string


    def displayTree(self, edges, nodes):
        """
        requires matplotlib and networkx which can be install by running 
        "pip install matplotlib" & "pip install networkx" 

        AND Graphviz MUST be installed onto computer
        https://www.graphviz.org/download/

        AND the Graphviz/Bin/ folder MUST be added
        to the system environment variables' PATH
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        G = nx.DiGraph()
        
        # create graph to use to generate image of graph
        # graph = {
        #   state_string: [state_strings that are children]
        # }
        graph = {}
        for stepsTaken in nodes.keys():
            graph[nodes[stepsTaken]] = []
        
        for edge in edges:
            start, end = edge
            startNode, endNode = nodes[start], nodes[end]
            graph[startNode].append(endNode)

        for node in graph.keys():
            G.add_node(node)
            for child in graph[node]:
                G.add_edge(node, child)

        # same layout using matplotlib with no labels
        plt.title('Board states')
        pos=nx.nx_pydot.graphviz_layout(G, prog='dot')
        nx.draw(G, pos, with_labels=True, arrows=False, node_size=0)
        plt.show()
        time.sleep(10)
        plt.close()

def printPath(path):
    print(path)
  
if __name__ == "__main__":
    board = Board() # predefined board state, 4*4
    #board = Board(None, None, 3) # randomized 3x3
    board.printBoard()
    
    print('\nBFS:')
    start = time.time()
    printPath(board.solveWithBFS())
    end = time.time()
    print('Ran in ' + str(end - start) + ' seconds.')
    

    print('\nA* search with Hamming distance as heuristic:')
    start = time.time()
    printPath(board.solveWithHamming())
    end = time.time()
    print('Ran in ' + str(end - start) + ' seconds.')
    
    print('\nA* search with Manhattan distance as heuristic:')
    start = time.time()
    printPath(board.solveWithManhattan())
    end = time.time()
    print('Ran in ' + str(end - start) + ' seconds.')

    board = Board([2, 8, 3, 1, 6, 4, 7, float('inf'), 5]) # board state from slide

    # ************** requirements MUST be installed to use the follow methods ****************
    #            requirements are listed inside of functions at bottom of class
    #board.BFS_likeSlide()
    