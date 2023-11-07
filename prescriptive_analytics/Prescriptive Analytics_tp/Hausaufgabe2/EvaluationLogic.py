import numpy 
"""注意：这个类没有构造函数，因此在构造EvaluationLogic对象的时候，没有任何函数会自动执行！！！"""

# 没有构造函数意味着，每当我们构造这个类的对象时，要想实现某种功能，就必须调用成员函数
# # 无论是currentJob还是firstJob，都是outputJob类。
# 这个类的最终目的就是计算当前方案current Solution的总时长Makespan！！！
# 前8章还将了怎么在这个类中计算订单延期
class EvaluationLogic:    

    # 这个成员函数需要一个Solution类的对象作为参数
    def DefineStartEnd(self, currentSolution):    
        #####
        # schedule first job: starts when finished at previous stage
        firstJob = currentSolution.OutputJobs[currentSolution.Permutation[0]]

        # firstJob.EndTimes=  是在往0组成的list中填入累积时间
        firstJob.EndTimes = numpy.cumsum([firstJob.ProcessingTime(x) for x in range(len(firstJob.EndTimes))]) # 后半部分的len(.endtimes)调用的还是由0组成的list，此处仅仅计算元素个数
        firstJob.StartTimes[1:] = firstJob.EndTimes[:-1]
        # firstjob开始时间默认是0，因此starttimes从第二项开始赋值，第二个机器上开始工作的时间是第一个机器结束的时间


        #####
        
        # schedule further jobs: starts when finished at previous stage and the predecessor is no longer on the considered machine
        for j in range(1,len(currentSolution.Permutation)): # 上面几行把0号也就是第一个job的开始时间和结束时间列表做好了，下面从1号第二个job开始指定开始结束时间
            currentJob = currentSolution.OutputJobs[currentSolution.Permutation[j]]# 当前job是permutation中的第j个
            previousJob = currentSolution.OutputJobs[currentSolution.Permutation[j-1]]# j-1没有顾虑，因为j从1 开始

            # first machine
            currentJob.StartTimes[0] = previousJob.EndTimes[0] # Starttimes的首个元素就是在第一个机器上开始处理的时间
            currentJob.EndTimes[0] = currentJob.StartTimes[0] + currentJob.ProcessingTime(0)

            # other machines
            for i in range(1,len(currentJob.StartTimes)):
                currentJob.StartTimes[i] = max(previousJob.EndTimes[i], currentJob.EndTimes[i-1]) # 当前job在下一台机器的开始时间取决于两方面，1：当前job在上一台机器的结束时间和2: 前一个job在当前机器的结束时间
                currentJob.EndTimes[i] = currentJob.StartTimes[i] + currentJob.ProcessingTime(i) # 当前job的结束时间等于当前job在当前机器的开始时间加上处理时间
        #####
        # Save Makespan and return Solution
        currentSolution.Makespan = currentSolution.OutputJobs[currentSolution.Permutation[-1]].EndTimes[-1]

    def CalcuateTardyJobs(solution):
        late_jobs = []
        for job in solution.OutputJobs.values():
            finish = job.EndTimes[-1]

            if finish > job.DueDate:
               late_jobs.append(job)
        return len(late_jobs)
