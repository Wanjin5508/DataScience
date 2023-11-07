from Neighborhood import *
import math
from copy import deepcopy
import numpy

""" Base class for several types of improvement algorithms. """ 
class ImprovementAlgorithm:
    def __init__(self, inputData, neighborhoodEvaluationStrategy = 'BestImprovement', neighborhoodTypes = ['Swap']):
        self.InputData = inputData

        self.EvaluationLogic = {}
        self.SolutionPool = {}
        self.RNG = {}

        self.NeighborhoodEvaluationStrategy = neighborhoodEvaluationStrategy
        self.NeighborhoodTypes = neighborhoodTypes
        self.Neighborhoods = {}

    def Initialize(self, evaluationLogic, solutionPool, rng = None):
        self.EvaluationLogic = evaluationLogic
        self.SolutionPool = solutionPool
        self.RNG = rng

    """ Similar to the so-called factory concept in software design. """
    def CreateNeighborhood(self, neighborhoodType, bestCurrentSolution):
        if neighborhoodType == 'Swap':
            return SwapNeighborhood(self.InputData, bestCurrentSolution.Permutation, self.EvaluationLogic, self.SolutionPool)
        elif neighborhoodType == 'Insertion':
            return InsertionNeighborhood(self.InputData, bestCurrentSolution.Permutation, self.EvaluationLogic, self.SolutionPool)
        elif neighborhoodType == 'TwoEdgeExchange':
            return TwoEdgeExchangeNeighborhood(self.InputData, bestCurrentSolution.Permutation, self.EvaluationLogic, self.SolutionPool)
        else:
            raise Exception(f"Neighborhood type {neighborhoodType} not defined.")

    def InitializeNeighborhoods(self, solution):
        for neighborhoodType in self.NeighborhoodTypes:
            neighborhood = self.CreateNeighborhood(neighborhoodType, solution)
            self.Neighborhoods[neighborhoodType] = neighborhood

""" Iterative improvement algorithm through sequential variable neighborhood descent. """
class IterativeImprovement(ImprovementAlgorithm):
    def __init__(self, inputData, neighborhoodEvaluationStrategy = 'BestImprovement', neighborhoodTypes = ['Swap']):
        super().__init__(inputData, neighborhoodEvaluationStrategy, neighborhoodTypes)

    def Run(self, solution):
        self.InitializeNeighborhoods(solution)    

        # According to "Hansen et al. (2017): Variable neighorhood search", this is equivalent to the 
        # sequential variable neighborhood descent with a pipe neighborhood change step.
        for neighborhoodType in self.NeighborhoodTypes:
            neighborhood = self.Neighborhoods[neighborhoodType]

            neighborhood.LocalSearch(self.NeighborhoodEvaluationStrategy, solution)
        
        return solution
        