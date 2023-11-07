class EvaluationLogic:
    # def __init__(self, inputData):
    #     self.InputData = inputData
    #     self.TotalTransport = None

    def DefineTotalTransport(self, currentSolution, withMinimumDistance = False):
        # 这个函数需要再solver中调用，并且能同时输出考虑MinimumDistance和不考虑的结果
        if withMinimumDistance == False:
            totalTransport = (currentSolution.TransportMatrix * currentSolution.ArrangedDMatrix).sum()
        else:
            totalTransport = (currentSolution.TransportMatrix * currentSolution.UpdatedDMatrix).sum()
        currentSolution.TotalTransport = totalTransport

        return totalTransport
