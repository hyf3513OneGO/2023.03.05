import datetime
import os

import numpy as np

from utils.datasetUtils import loadDataset
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
'''
MECInfo = [196, 0.2, 10]  # f,k,B
MECInfo = np.array(MECInfo)

n = 10
B = [0.8, 0.2] * n
B = np.array(B)
B = B.reshape(n, 2)

randTasksTimeMats = getRandomDAGMats(n, 6, 10)
randTasksDataMats = getRandomDataMat(randTasksTimeMats, 5, 8)
randomDecision = getRandomDecisionMats(randTasksTimeMats.copy(), False, [0, 2])
randomResource = getRandomDecisionMats(randTasksTimeMats.copy(), True, [1, 10]) * randomDecision
randomCarInfos = getRandomCarInfos(n)

MECInfo = [196, 0.02, 10]  # f,k,B
MECInfo = np.array(MECInfo)
ITER = 50

datasetDir = "./datasets/2023.03.07"


def main():
    ESTGraphs = []
    timeDataset, dataDataset, decisionDataset, resourceDataset, carDataset = loadDataset(os.path.join(datasetDir))
    timeMat = timeDataset[0]
    dataMat = dataDataset[0]
    decisionMat = decisionDataset[0]
    resourceMat = resourceDataset[0]
    carMat = carDataset[0]
    taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal, taskTimeOriginTotal, taskEnergyOriginTotal = calcTasks(
        timeMat.copy(),
        dataMat.copy(),
        decisionMat.copy(),
        resourceMat.copy(),
        carMat.copy(), MECInfo.copy(),
        B.copy())
    print("原始数据：")
    origin = [taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal]
    print(origin)
    decisionMatResult = decisionDataset[0].copy()
    resourceMatResult = resourceDataset[0].copy()

    timeList = []
    energyList = []
    transferList = []
    QOEList = []
    timeList.append(taskTimeTotal)
    energyList.append(taskEnergyTotal)
    transferList.append(transferTimeTotal)
    QOEList.append(QOETotal)
    opts = []
    for i in range(ITER):
        timeNow = datetime.datetime.now()
        print("----------------------------------------------------------------------------------")
        print(f"正在迭代{i + 1}次,{timeNow.strftime('%Y-%m-%d %H:%M:%S %f')}")

        decisionMatResult = gaResolve_class_decision(timeMat.copy(),
                                                     dataMat.copy(),
                                                     decisionMatResult,
                                                     resourceMatResult,
                                                     carMat.copy(), MECInfo.copy(),
                                                     B.copy())
        # resourceMatResult = gaResolve_class_resource(randTasksTimeMats.copy(), randTasksDataMats.copy(),
        #                                              decisionMatResult.copy(), resourceMatResult.copy(),
        #                                              randomCarInfos.copy(), MECInfo.copy(), B.copy())
        resourceMatResult = psoResolve_class_resource(timeMat.copy(),
                                                      dataMat.copy(),
                                                      decisionMatResult,
                                                      resourceMatResult,
                                                      carMat.copy(), MECInfo.copy(),
                                                      B.copy())
        taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal, taskTimeOriginTotal, taskEnergyOriginTotal = calcTasks(
            timeMat.copy(),
            dataMat.copy(),
            decisionMatResult,
            resourceMatResult,
            carMat.copy(), MECInfo.copy(),
            B.copy())
        timeDone = datetime.datetime.now()
        timeDelta = timeDone - timeNow
        print(f"{i + 1}次的结果,{timeDone.strftime('%Y-%m-%d %H:%M:%S %f')},耗时:{timeDelta.seconds}秒")
        print(taskTimeTotal, taskEnergyTotal, transferTimeTotal, QOETotal)
        if QOEList[-1] == QOETotal:
            break
        timeList.append(taskTimeTotal)
        energyList.append(taskEnergyTotal)
        transferList.append(transferTimeTotal)
        QOEList.append(QOETotal)

    print("--------------------------------------Result------------------------------------------")
    resultList = [timeList, energyList, transferList, QOEList]
    allEvaluate(resultList)
    plotGraph(resultList)


if __name__ == "__main__":
    main()
