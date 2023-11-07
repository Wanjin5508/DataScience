from InputData import *
from OutputData import *
from ConstructiveHeuristic import *
from ImprovementAlgorithm import *
from EvaluationLogic import *

import random

class Solver:
    def __init__(self, inputData, seed):
        self.InputData = inputData
        self.Seed = seed
        self.RNG = numpy.random.default_rng(self.Seed)
        self.EvaluationLogic = EvaluationLogic(inputData)
        self.SolutionPool = SolutionPool()
        
        self.ConstructiveHeuristic = ConstructiveHeuristics(self.EvaluationLogic, self.SolutionPool)      

    def Initialize(self):
        self.OptimizationAlgorithm.Initialize(self.EvaluationLogic, self.SolutionPool, self.RNG)
    
    def ConstructionPhase(self, constructiveSolutionMethod):
        self.ConstructiveHeuristic.Run(self.InputData, constructiveSolutionMethod)

        bestInitalSolution = self.SolutionPool.GetLowestMakespanSolution()

        print("Constructive solution found.")
        print(bestInitalSolution)

        return bestInitalSolution

    def ImprovementPhase(self, startSolution, algorithm):
        algorithm.Initialize(self.EvaluationLogic, self.SolutionPool, self.RNG)
        bestSolution = algorithm.Run(startSolution)

        print("Best found Solution.")
        print(bestSolution)

    def RunLocalSearch(self, constructiveSolutionMethod, algorithm):
        startSolution = self.ConstructionPhase(constructiveSolutionMethod)

        self.ImprovementPhase(startSolution, algorithm)


if __name__ == '__main__':  ##???????????????????????????????????????????????????????????

    data = InputData("InputFlowshopSIST.json")

    localSearch = IterativeImprovement(data, 'BestImprovement', ['Swap'])
    
    solver = Solver(data, 1008)

    solver.RunLocalSearch(
        constructiveSolutionMethod='FCFS',
        algorithm=localSearch)
