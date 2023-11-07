from Solver import *
from InputData import *


data = InputData("InputFlowshopSIST20.json")

localSearch = IterativeImprovement(data, 'BestImprovement', ['TwoEdgeExchange'])

solver = Solver(data, 1008)

solver.RunLocalSearch(
    constructiveSolutionMethod='FCFS',
    algorithm=localSearch)