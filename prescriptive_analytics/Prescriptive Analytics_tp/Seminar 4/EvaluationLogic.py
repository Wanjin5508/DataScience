import numpy 
from copy import deepcopy
class EvaluationLogic:
    def __init__(self, inputData):
        self.InputData = inputData

    def DefineStartEnd(self, currentSolution):    
        #####
        for position, jobId in enumerate(currentSolution.Permutation):
            currentJob = currentSolution.OutputJobs[jobId]
            if position == 0:
            # schedule first job: starts when finished at previous stage
                currentJob.EndTimes = numpy.cumsum([currentJob.ProcessingTime(x) for x in range(len(currentJob.EndTimes))])
                currentJob.StartTimes[1:] = currentJob.EndTimes[:-1]

            # schedule further jobs: starts when finished at previous stage and the predecessor is no longer on the considered machine
            else:
                # first machine
                currentJob.StartTimes[0] = previousJob.EndTimes[0]
                currentJob.EndTimes[0] = currentJob.StartTimes[0] + currentJob.ProcessingTime(0)
                # other machines
                for i in range(1,len(currentJob.StartTimes)):
                    currentJob.StartTimes[i] = max(previousJob.EndTimes[i], currentJob.EndTimes[i-1])
                    currentJob.EndTimes[i] = currentJob.StartTimes[i] + currentJob.ProcessingTime(i)
            previousJob = currentJob
            
        #####
        # Save Makespan and return Solution
        currentSolution.Makespan = currentSolution.OutputJobs[currentSolution.Permutation[-1]].EndTimes[-1]

    def CalculateTardiness(self, currentSolution):
        totalTardiness = 0
        for key in currentSolution.OutputJobs:
            if(currentSolution.OutputJobs[key].EndTimes[-1] - currentSolution.OutputJobs[key].DueDate > 0):
                currentSolution.OutputJobs[key].Tardiness = currentSolution.OutputJobs[key].EndTimes[-1] - currentSolution.OutputJobs[key].DueDate
                totalTardiness += currentSolution.OutputJobs[key].EndTimes[-1] - currentSolution.OutputJobs[key].DueDate
        currentSolution.TotalTardiness = totalTardiness       

    def CalculateWeightedTardiness(self, currentSolution):
        totalWeightedTardiness = 0
        for key in currentSolution.OutputJobs:
            if(currentSolution.OutputJobs[key].EndTimes[-1] - currentSolution.OutputJobs[key].DueDate > 0):
                currentSolution.OutputJobs[key].Tardiness = currentSolution.OutputJobs[key].EndTimes[-1] - currentSolution.OutputJobs[key].DueDate
                totalWeightedTardiness += (currentSolution.OutputJobs[key].EndTimes[-1] - currentSolution.OutputJobs[key].DueDate) * currentSolution.OutputJobs[key].TardCost
        currentSolution.TotalWeightedTardiness = totalWeightedTardiness

    def DetermineBestInsertion(self, solution, jobToInsert):
        ###
        # insert job at front of permutation
        solution.Permutation.insert(0, jobToInsert)
        bestPermutation = deepcopy(solution.Permutation)
        
        self.DefineStartEnd(solution)
        bestCmax = solution.Makespan

        ###
        # swap job i to each position and check for improvement
        lengthPermutation = len(solution.Permutation) - 1
        for j in range(0, lengthPermutation):
            solution.Permutation[j], solution.Permutation[j + 1] = solution.Permutation[j+1], solution.Permutation[j]
            self.DefineStartEnd(solution)
            if(solution.Makespan < bestCmax):
                bestCmax = solution.Makespan
                bestPermutation = [x for x in solution.Permutation]

        solution.Makespan = bestCmax
        solution.Permutation = bestPermutation
