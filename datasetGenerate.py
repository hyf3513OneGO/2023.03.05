import datetime
import os.path

import numpy as np

from utils.evaluateUtils import allEvaluate, plotGraph
from utils.calcUtils import calcTasks
from utils.mockUtils import getRandomDAGMats, getRandomDecisionMats, getRandomCarInfos, \
    getRandomDataMat
from utils.resolvers.decisions.ga import gaResolve_class_decision
from utils.resolvers.resources.pso import psoResolve_class_resource

'''
DAG最大节点数：6
单任务矩阵6x6
最大车辆数10
数据集为50次决策事件
'''
n = 10
B = [0.8, 0.2] * n
B = np.array(B)
B = B.reshape(n, 2)
ITER = 20
datasetDir = "./datasets/2023.03.07"
MECInfo = [196, 0.02, 10]  # f,k,B
MECInfo = np.array(MECInfo)


def main(size: int):
    datasetTimeList = []
    datasetDataList = []
    datasetCarInfoList = []
    datasetRandomDecisionList = []
    datasetRandomResourceList = []
    for i in range(size):
        randTasksTimeMats = getRandomDAGMats(n, 6, 10)
        randTasksDataMats = getRandomDataMat(randTasksTimeMats, 5, 8)
        randomDecision = getRandomDecisionMats(randTasksTimeMats.copy(), False, [0, 2])
        randomResource = getRandomDecisionMats(randTasksTimeMats.copy(), True, [1, 10]) * randomDecision
        randomCarInfos = getRandomCarInfos(n)
        datasetTimeList.append(randTasksTimeMats)
        datasetDataList.append(randTasksDataMats)
        datasetCarInfoList.append(randomCarInfos)
        datasetRandomDecisionList.append(randomDecision)
        datasetRandomResourceList.append(randomResource)
    np.save(os.path.join(datasetDir, "time.npy"), np.stack(datasetTimeList))
    np.save(os.path.join(datasetDir, "data.npy"), np.stack(datasetDataList))
    np.save(os.path.join(datasetDir, "car.npy"), np.stack(datasetCarInfoList))
    np.save(os.path.join(datasetDir, "decision.npy"), np.stack(datasetRandomDecisionList))
    np.save(os.path.join(datasetDir, "resource.npy"), np.stack(datasetRandomResourceList))



if __name__ == "__main__":
    main(50)
