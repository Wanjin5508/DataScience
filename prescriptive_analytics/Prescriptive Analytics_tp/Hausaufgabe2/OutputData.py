import sys
import csv

from InputData import *

# 定义一个DataJob类的子类
class OutputJob(DataJob):

    # 该子类的构造函数需要DataJob类的对象作为参数，并用这个对象的属性进行计算
    def __init__(self, dataJob):
        # 调用父类的构造函数构造出DataJob对象
        super().__init__(dataJob.JobId, [dataJob.ProcessingTime(i) for i in range(len(dataJob.Operations))], 
                        [dataJob.SetupTime(i) for i in range(len(dataJob.Operations))], dataJob.DueDate, dataJob.TardCost)
        
        # 子类的对象在父类基础上多了下面几个属性, 全部初始化为0，若属性为列表，则初始化为有0组成的列表，列表共有n个元素。
        # 记住这些属性！别跟Solution类的属性混淆。
        self.StartSetups = [0]*len(self.Operations) # 全部元素被初始化为0.Operations是每个job被处理的时间长度组成的list
        self.EndSetups = [0]*len(self.Operations)        
        self.StartTimes = [0]*len(self.Operations)
        self.EndTimes = [0]*len(self.Operations)
        self.Tardiness = 0

# 看构造函数的参数表，需要一个订单列表和所有订单的排列，才能构造出一个solution对象
class Solution:
    def __init__(self, jobList, permutation):  # jobList实际上是InputData类的对象的InputJobs属性，是个列表
        self.OutputJobs = {}

        # 遍历jobList，jobId是列表的索引，job是一个DataJob类的订单
        for jobId, job in enumerate(jobList):
            # 用DataJob类的对象作为参数，构造出OutputJob的对象，并把OutputJob对象加入字典中
            self.OutputJobs[jobId] = OutputJob(job)
        self.Permutation = permutation
        self.Makespan = -1
        self.TotalTardiness = -1
        self.TotalWeightedTardiness = -1

    # Solution对象的输出，重点在订单排列和总时长Makespan。Makespan需要经过EvaluationLogic才能被赋值。详情见前八章
    def __str__(self):
        return "The permutation " + str(self.Permutation) + " results in a Makespan of " + str(self.Makespan)

    # 这个成员函数用于自定一个排列方式
    def SetPermutation(self, permutation):
        self.Permutation = permutation

    def WriteSolToCsv(self, fileName):
        csv_file = open(fileName, "w")
        csv_file.write('Machine,Job,Start_Setup,End_Setup,Start,End\n')
        for job in self.OutputJobs.values():
            for i in range(len(job.EndTimes)):
                csv_file.write(str(i + 1) + "," + str(job.JobId) + "," + str(job.StartSetups[i]) + "," 
                            + str(job.EndSetups[i]) + "," + str(job.StartTimes[i]) + "," + str(job.EndTimes[i]) + "\n")
