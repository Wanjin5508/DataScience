from InputData import *
from OutputData import *
from ConstructiveHeuristic import *
from ImprovementAlgorithm import *
from EvaluationLogic import *

import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

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

    def EvalPFSP(self, individual):
        solution = Solution(self.InputData.InputJobs, individual)
        self.EvaluationLogic.DefineStartEnd(solution)
        return [solution.Makespan]

    def RunGeneticAlgorithm(self, populationSize, generations, matingProb, mutationProb):
        # Creator - meta-factory to create new classes
        creator.create("FitnessMin", base.Fitness, weights=[-1.0])
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox() 

        # Individual is a permutation of integer indices
        toolbox.register("indices", solver.RNG.choice, range(self.InputData.n), self.InputData.n, replace=False)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)

        # Population is a collection of individuals
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        pop = toolbox.population(populationSize) # number of individuals in population

        # Operators
        toolbox.register("mate", tools.cxPartialyMatched) # set crossover 
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05) # set mutation
        toolbox.register("select", tools.selTournament, tournsize=3) # set selection mechanism
        
        # Fitness function
        toolbox.register("evaluate", self.EvalPFSP) 
        
        # Statistics during run time
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
                
        # Hall of fame --> Best individual
        hof = tools.HallOfFame(1)

        random.seed(self.Seed)
        algorithms.eaSimple(population=pop, toolbox=toolbox, cxpb=matingProb, mutpb=mutationProb, ngen=generations, stats=stats, halloffame=hof)

        bestPermutation = hof[0]
        bestSolution = Solution(self.InputData.InputJobs, bestPermutation)
        self.EvaluationLogic.DefineStartEnd(bestSolution)
        print(f'Best found Solution.\n {bestSolution}')


if __name__ == '__main__':

    data = InputData("TestInstancesJson/Large/VFR100_20_1_SIST.json") # TestInstances/Small/VFR40_10_3_SIST.txt 

    insertionLocalSearch = IterativeImprovement(data, 'FirstImprovement', ['TaillardInsertion'])
    iteratedGreedy = IteratedGreedy(
    data, 
    numberJobsToRemove=2, 
    baseTemperature=1, 
    maxIterations=10, 
    localSearchAlgorithm=insertionLocalSearch
    )

    solver = Solver(data, 1008)

    print('Start IG\n')
    solver.RunLocalSearch(
        constructiveSolutionMethod='NEH',
        algorithm=iteratedGreedy)
    
    print('Start GA\n')
    solver.RunGeneticAlgorithm(
        populationSize=100, 
        generations=200, 
        matingProb=0.8, 
        mutationProb=0.2)
