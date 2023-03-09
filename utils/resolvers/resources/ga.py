import time

import geatpy as ea
import numpy as np

# decisionMat不是决策变量的矩阵而是子任务的矩阵
from utils.calcUtils import calcTasks
from utils.mockUtils import getRandomDAGMats, getRandomDataMat, getRandomDecisionMats, getRandomCarInfos
from utils.solveUtils import mat2geatVar, geatVars2mat


class MyProblem(ea.Problem):  # 继承Problem父类
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
        self.B = B
        self.Parallel =True
        self.pm=0.7
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Chrom = pop.Chrom  # 获取决策变量矩阵，它等于种群的表现型矩阵Phen
        cvList = []
        QOElist = []
        for idx, item in enumerate(Chrom):
            cvList.append([0])
            decisionMat, _ = geatVars2mat(Chrom[idx], self.taskTimeMats)
            taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal,taskTimeOriginTotal,taskEnergyOriginTotal = calcTasks(self.taskTimeMats,
                                                                                    self.taskDataMats,
                                                                                    self.decisionsMats, decisionMat,
                                                                                    self.carInfos, self.MECInfo, self.B)
            QOElist.append([QOETotal])
        pop.CV = np.stack(cvList)
        pop.ObjV = np.stack(QOElist)
        # print(pop.FitnV)
        # print(pop.ObjV)
        # print(pop.CV)
        # FitnV = ea.ranking(ObjV, CV, maxormins)


def gaResolve_class_resource(taskTimeMats, tasksDataMats, decisionMat, ResourceMats, CarInfos, MECInfo, B):
    # 实例化问题对象
    problem = MyProblem(taskTimeMats, tasksDataMats, decisionMat, ResourceMats, CarInfos, MECInfo, B)
    # 构建算法
    algorithm = ea.soea_SGA_templet(problem,
                                            ea.Population(Encoding='RI', NIND=50),
                                            MAXGEN=10,  # 最大进化代数
                                            logTras=1)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    # 求解
    res = ea.optimize(algorithm, seed=1, verbose=False, drawing=0, outputMsg=False, drawLog=False, saveFlag=False,
                      dirName='result')
    decisionMatResult,_ = geatVars2mat(np.squeeze(res['Vars']), taskTimeMats)
    return decisionMatResult
