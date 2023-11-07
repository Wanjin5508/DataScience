import sys
import csv

from InputData import *

class OutputJob(DataJob):

    def __init__(self, dataJob):
        super().__init__(dataJob.JobId, [dataJob.ProcessingTime(i) for i in range(len(dataJob.Operations))], [dataJob.SetupTime(i) for i in range(len(dataJob.Operations))], dataJob.DueDate, dataJob.TardCost)
        
        self.StartSetups = [0]*len(self.Operations)
        self.EndSetups = [0]*len(self.Operations)        
        self.StartTimes = [0]*len(self.Operations)
        self.EndTimes = [0]*len(self.Operations)
        self.Tardiness = 0

