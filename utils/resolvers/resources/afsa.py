import time

from matplotlib import pyplot as plt
from sko.AFSA  import AFSA
import numpy as np

# decisionMat不是决策变量的矩阵而是子任务的矩阵
from sko.tools import set_run_mode

from utils.calcUtils import calcTasks
from utils.mockUtils import getRandomDAGMats, getRandomDataMat, getRandomDecisionMats, getRandomCarInfos
from utils.solveUtils import mat2geatVar, geatVars2mat

max_iter = 10


class MyProblem:
    def __init__(self, taskTimeMats, taskDataMats, decisionsMats, resourceMats, carInfos, MECInfos, B):
        geatVars, lenV = mat2geatVar(taskTimeMats)
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 优化目标个数
        maxormins = [-1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = lenV  # 初始化Dim（决策变量维数）
        varTypes = [0 for _ in range(lenV)]  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [1 for _ in range(lenV)]  # 决策变量下界
        ub = [10 for _ in range(lenV)]  # 决策变量上界
        lbin = [1 for _ in range(lenV)]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1 for _ in range(lenV)]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        self.taskTimeMats = taskTimeMats
        self.taskDataMats = taskDataMats
        self.decisionsMats = decisionsMats
        self.resourceMats = resourceMats
        self.carInfos = carInfos
        self.MECInfo = MECInfos
        self.taskTimeOriginTotal = 0
        self.B = B
        self.Parallel = True
        self.pm = 0.7
        set_run_mode(self.aimFunc, 'cached')

    def aimFunc(self, Chrom):
        decisionMat, _ = geatVars2mat(Chrom, self.taskTimeMats)
        taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal, taskTimeOriginTotal, taskEnergyOriginTotal = calcTasks(
            self.taskTimeMats,
            self.taskDataMats,
            self.decisionsMats, decisionMat,
            self.carInfos, self.MECInfo, self.B)
        self.taskTimeOriginTotal = taskTimeOriginTotal
        return -QOETotal

    def checkTimeOpT(self, Chrom):
        # print(Chrom)
        decisionMat, _ = geatVars2mat(Chrom, self.taskTimeMats)
        taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal, taskTimeOriginTotal, taskEnergyOriginTotal = calcTasks(
            self.taskTimeMats,
            self.taskDataMats,
            self.decisionsMats, decisionMat,
            self.carInfos, self.MECInfo, self.B)
        return taskTimeTotal - taskTimeOriginTotal

    def solve(self):
        geatVars, lenV = mat2geatVar(self.taskTimeMats)
        lb = [1 for _ in range(lenV)]  # 决策变量下界
        ub = [10 for _ in range(lenV)]  # 决策变量上界
        # lbin = [1 for _ in range(lenV)]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        # ubin = [1 for _ in range(lenV)]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        afsa = AFSA (func=self.aimFunc, n_dim=lenV, size_pop=40, max_iter=max_iter)
        afsa.run()
        decisionMatResult, _ = geatVars2mat(afsa.best_x, self.taskTimeMats)
        # print(decisionMatResult)
        return decisionMatResult
        # plt.plot(pso.gbest_y_hist)
        # plt.show()


def afsaResolve_class_resource(taskTimeMats, tasksDataMats, decisionMat, ResourceMats, CarInfos,
                              MECInfo, B):
    problem = MyProblem(taskTimeMats, tasksDataMats, decisionMat, ResourceMats, CarInfos, MECInfo, B)
    return problem.solve()
