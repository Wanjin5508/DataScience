import json

# 机器类，功能很简单，构造函数只需要机器的id。__str__()函数用于输出 Machine...。需要注意被修饰器修饰的函数，这个函数可以当作属性来调用（调用函数时的括号可以省略）
class DataMachine:

    def __init__(self, machineId):
        self.__machId = machineId

    def __str__(self):
        result = "Machine " + str(self.__machId)

        return result
        
    @property
    def MachineId(self):
        return self.__machId

# 这个类用于构造一个订单Job对象，构造函数的参数表来自json文件中的key
class DataJob:
    
    def __init__(self, idJob, processingTimes, setupTimes, dueDate, tardinessCost):
        self.__jobId = idJob
        self.__processingTimes = processingTimes
        self.__dueDate = dueDate
        self.__setupTimes = setupTimes
        self.__tardCost = tardinessCost
        
    # 用于打印对象时自动调用，表达某个订单需要在多少台机器上处理（operations），以及每台机器处理该订单的时长
    def __str__(self):
        result = f"Job {self.__jobId} with {len(self.__processingTimes)} Operations:\n" # 一个订单共需要几次操作？？

        for opId, processingTime in enumerate(self.__processingTimes):
            result += f"Operation {opId} with Processingtime: {processingTime} \n"  # 每台机器上各需要加工多久？？？

        return result  # __str__函数一定要有返回值！！
    
    # DataJob的5个属性再加上一个Operation（操作次数）都以成员函数的形式给出。注意这里分两种情况，
    # 1.成员函数的参数表只有self，不需要额外的参数。也就是说一个DataJob对象完全可以利用自身属性作为该成员函数的返回值。
    #   这种情况需要使用修饰器，这样我们一方面可以像调用属性那样调用函数，比较方便。另一方面，我们的属性是以私有private形式定义的，在其他类中无法调用，
    #   因此需要把属性定义成函数的形式，在使用修饰器的情况下，在调用的时候就可以假装属性是公有的
    # 2.成员函数有其他的参数，这种情况下就不适合用修饰器，而是要定义成普通的成员函数，这样这个类的对象就可以随时调用该函数。
    #   比如我想知道一个订单job在第一台机器上的加工时间，可以用job.ProcessingTime(1)
    @property
    def JobId(self):
        return self.__jobId

    @property
    def TardCost(self):
        return self.__tardCost
        
    @property
    def DueDate(self):
        return self.__dueDate

    @property
    def Operations(self): ## 注意operations这个特殊的形式，opID就是机器id
        return [(opId, processingTime) for opId, processingTime in enumerate(self.__processingTimes)]
    
    def SetupTime(self, position):
        return self.__setupTimes[position]

    def ProcessingTime(self, position):
        return self.__processingTimes[position]

# 这个类用于导入json数据，json中包含项目名称，机器数量m，订单数量n，所有订单的详细信息InputJobs。注意类的属性，尤其是InputJobs
class InputData:

    def __init__(self, path):
        self.__path = path
        self.DataLoad()  # 构造方法中调用成员函数，类似习题1，即在构造对象时完成成员函数的调用。对象不需要再次调用成员函数

    def DataLoad(self):

        with open(self.__path, "r") as inputFile:
            inputData = json.load(inputFile)  # inputData是个字典！！
        
        self.n = inputData['nJobs']
        self.m = inputData['nMachines']
        
        self.InputJobs = list()
        self.InputMachines = list()

        for job in inputData['Jobs']:  # inputData['Jobs']是一个列表，其元素job是字典。
            # 下一行构造DataJob对象，并把对象加到InputJobs中  --> InputJobs是个由DataJob对象组成的list，而DataJob类有__str__函数，因此InputJobs中的元素可以遍历后打印
            self.InputJobs.append(DataJob(job['Id'], job['ProcessingTimes'], job["SetupTimes"], job['DueDate'], job['TardCosts']))
        
        for k in range(self.m):
            self.InputMachines.append(DataMachine(k))